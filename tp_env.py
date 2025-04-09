import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from numpy.linalg import solve, matrix_rank
from numpy import pi, cos, sin, hstack, zeros, ones
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
import control
from reward import RewardManager

class TriplePendulumEnv:
    def __init__(self, reward_manager: RewardManager = None, render_mode=None, num_nodes=3, arm_length=1./3, bob_mass=0.01/3, friction_coefficient=0.1):
        self.reward_manager = reward_manager
        self.render_mode = render_mode
        self.n = num_nodes
        self.arm_length = arm_length
        self.bob_mass = bob_mass
        self.friction_coefficient = friction_coefficient

        # Paramètres de simulation
        self.dt = 0.01  # intervalle de temps
        self.current_state = None
        self.current_time = 0.0

        # Variables pour contrôler l’animation (si rendu)
        self.applied_force = 0.0
        self.force_increment = 0.3
        self.target_force = 0.0
        self.force_smoothing = 0.1
        self.current_frame = 0

        # Modèle symbolique
        self.q = dynamicsymbols(f'q:{num_nodes+1}')  # coordonnées généralisées
        self.u = dynamicsymbols(f'u:{num_nodes+1}')  # vitesses généralisées
        self.f = dynamicsymbols('f')                # force appliquée au chariot

        self.m = symbols(f'm:{num_nodes+1}')          # masses
        self.l = symbols(f'l:{num_nodes}')            # longueurs
        self.g, self.t = symbols('g t')

        self._setup_symbolic_model()
        self._setup_numeric_evaluation()

    def _setup_symbolic_model(self):
        I = ReferenceFrame('I')
        O = Point('O')
        O.set_vel(I, 0)

        P0 = Point('P0')
        P0.set_pos(O, self.q[0]*I.x)
        P0.set_vel(I, self.u[0]*I.x)
        Pa0 = Particle('Pa0', P0, self.m[0])

        frames = [I]
        points = [P0]
        particles = [Pa0]

        force_cart = self.f * I.x
        weight_cart = -self.m[0]*self.g*I.y
        friction_cart = -self.friction_coefficient * self.u[0]*I.x
        forces = [(P0, force_cart + weight_cart + friction_cart)]
        kindiffs = [self.q[0].diff(self.t) - self.u[0]]

        for i in range(self.n):
            Bi = I.orientnew(f'B{i}', 'Axis', [self.q[i+1], I.z])
            Bi.set_ang_vel(I, self.u[i+1]*I.z)
            frames.append(Bi)

            Pi = points[-1].locatenew(f'P{i+1}', self.l[i]*Bi.x)
            Pi.v2pt_theory(points[-1], I, Bi)
            points.append(Pi)

            Pai = Particle(f'Pa{i+1}', Pi, self.m[i+1])
            particles.append(Pai)

            weight = -self.m[i+1]*self.g*I.y
            friction = -self.friction_coefficient * self.u[i+1]*I.z
            forces.append((Pi, weight + friction))
            kindiffs.append(self.q[i+1].diff(self.t) - self.u[i+1])

        self.kane = KanesMethod(I, q_ind=self.q, u_ind=self.u, kd_eqs=kindiffs)
        self.fr = self.kane._form_fr(forces)
        self.frstar = self.kane._form_frstar(particles)

    def _setup_numeric_evaluation(self):
        # Paramètres numériques
        parameters = [self.g, self.m[0]]
        self.parameter_vals = [9.81, self.bob_mass]
        for i in range(self.n):
            parameters += [self.l[i], self.m[i+1]]
            self.parameter_vals += [self.arm_length, self.bob_mass]

        dynamic = self.q + self.u
        dynamic.append(self.f)
        dummy_symbols = [Dummy() for _ in dynamic]
        dummy_dict = dict(zip(dynamic, dummy_symbols))
        kindiff_dict = self.kane.kindiffdict()
        M = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
        F = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)
        self.M_func = lambdify(dummy_symbols + parameters, M)
        self.F_func = lambdify(dummy_symbols + parameters, F)

    def rhs(self, x, t, args, controller=None):
        # Calcul de la dérivée d'état
        u_input = controller(x) if controller else 0.0
        arguments = hstack((x, u_input, args))
        dx = np.array(solve(self.M_func(*arguments), self.F_func(*arguments))).T[0]
        return dx

    def reset(self):
        # Initialisation de l'état : position initiale du chariot, angles et vitesses faibles
        position_initial = 0.0
        angles_init = -pi/2
        speeds_init = 1e-3
        state = hstack((position_initial, angles_init*ones(len(self.q)-1), speeds_init*ones(len(self.u))))
        self.current_state = state.copy()
        self.current_time = 0.0
        return state

    def get_state(self):
        return self.current_state if self.current_state is not None else self.reset()

    def _render_init(self):
        # Méthode optionnelle d'initialisation du rendu
        pass

    def animate_pendulum(self, steps, states, length, title='Pendulum'):
        # Méthode d'affichage (identique à la version initiale)
        num_joints = states.shape[1] // 2
        fig = plt.figure(facecolor='#F0F0F5')
        cart_width, cart_height = 0.4, 0.2
        window_width = 4.0
        xmin = -window_width/2
        xmax = window_width/2
        ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')
        ax.set_title(title, color='#3C3C46', fontsize=12, pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        BACKGROUND_COLOR = '#F0F0F5'
        CART_COLOR = '#32323C'
        TRACK_COLOR = '#B4B4BE'
        PENDULUM_COLORS = ['#DC3C3C', '#3CB43C', '#3C3CDC']
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        for x_val in np.arange(xmin, xmax + 0.5, 0.5):
            ax.axvline(x=x_val, color='#D2D2D7', linestyle='-', alpha=0.3)
        for y_val in np.arange(-1, 1.1, 0.5):
            ax.axhline(y=y_val, color='#D2D2D7', linestyle='-', alpha=0.3)
        track = Rectangle([xmin, -cart_height/2 - 0.05], window_width, 0.05, facecolor=TRACK_COLOR, edgecolor='none')
        ax.add_patch(track)
        ax.plot([0], [-cart_height/2 - 0.025], '|', color='#64646E', markersize=10, markeredgewidth=2)
        time_display = ax.text(0.04, 0.9, '', transform=ax.transAxes, color='#3C3C46', fontsize=10)
        cart = Rectangle([states[0, 0]-cart_width/2, -cart_height/2], cart_width, cart_height, facecolor=CART_COLOR, edgecolor='none')
        ax.add_patch(cart)
        highlight = Rectangle([states[0, 0]-cart_width/2+0.02, -cart_height/2+0.02], cart_width-0.04, cart_height/3, facecolor='#50505A', edgecolor='none')
        ax.add_patch(highlight)
        pendulum_line, = ax.plot([], [], color=PENDULUM_COLORS[0],
                                   linewidth=4, marker='o', markersize=8,
                                   markeredgecolor='#1E1E28', markerfacecolor='#5A5A64')

        def handle_key_press(event):
            if event.key == 'left':
                self.target_force = -self.force_increment
            elif event.key == 'right':
                self.target_force = self.force_increment
            elif event.key == ' ':
                self.target_force = 0.0
                states[self.current_frame] = self.reset()
                for i in range(self.current_frame+1, len(states)):
                    states[i] = states[self.current_frame]

        def handle_key_release(event):
            if event.key in ['left', 'right']:
                self.target_force = 0.0

        fig.canvas.mpl_connect('key_press_event', handle_key_press)
        fig.canvas.mpl_connect('key_release_event', handle_key_release)

        def initialize_animation():
            time_display.set_text('')
            cart.set_xy((0.0, -cart_height/2))
            highlight.set_xy((0.0 - cart_width/2+0.02, -cart_height/2+0.02))
            pendulum_line.set_data([], [])
            return time_display, cart, highlight, pendulum_line

        def step(frame):
            self.current_frame = frame
            if frame < steps - 1:
                self.applied_force += self.force_smoothing*(self.target_force - self.applied_force)
                current_state = states[frame]
                time_step = self.dt
                next_state = current_state + self.rhs(current_state, frame*self.dt, self.parameter_vals, lambda x: self.applied_force)*time_step
                cart_position = next_state[0]
                if cart_position - cart_width/2 < xmin:
                    next_state[0] = xmin+cart_width/2
                    next_state[num_joints] = 0
                elif cart_position + cart_width/2 > xmax:
                    next_state[0] = xmax-cart_width/2
                    next_state[num_joints] = 0
                states[frame+1] = next_state

            time_display.set_text(f'time = {frame*self.dt:.2f}\nforce = {self.applied_force:.2f}')
            cart.set_xy((states[frame, 0]-cart_width/2, -cart_height/2))
            highlight.set_xy((states[frame, 0]-cart_width/2+0.02, -cart_height/2+0.02))
            x_positions = hstack((states[frame, 0], zeros(num_joints-1)))
            y_positions = zeros(num_joints)
            for joint in range(1, num_joints):
                x_positions[joint] = x_positions[joint-1]+length*cos(states[frame, joint])
                y_positions[joint] = y_positions[joint-1]+length*sin(states[frame, joint])
            current_angle = states[frame, 1]
            color_index = min(2, int(abs(current_angle)/(pi/2)))
            pendulum_line.set_color(PENDULUM_COLORS[color_index])
            pendulum_line.set_data(x_positions, y_positions)
            return time_display, cart, highlight, pendulum_line

        initialize_animation()
        for frame in range(steps):
            step(frame)
            plt.pause(0.02)

if __name__ == "__main__":
    env = TriplePendulumEnv()
    state = env.reset()
    states = [state]
    max_steps = 500
    for _ in range(max_steps):
        action = 0.0  # Action nulle par défaut lors de l'initialisation
        dx = env.rhs(state, env.current_time, env.parameter_vals, lambda x: action)
        env.current_time += env.dt
        state = state + dx*env.dt
        states.append(state)
    states = np.vstack(states)
    while True:
        env.animate_pendulum(steps=max_steps, states=states, length=env.arm_length, title='Simulation step-by-step')
