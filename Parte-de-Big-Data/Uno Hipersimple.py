import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict


class UnoSimplificado(gym.Env):
    def __init__(self):
        super(UnoSimplificado, self).__init__()
        self.colores = ['R', 'A', 'V', 'Z']
        self.numeros = list(range(10))
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]
        # Acciones: 0: Jugar por Color, 1: Jugar por Número, 2: Robar
        self.action_space = spaces.Discrete(3)

    def _get_state(self):
        color_mesa, num_mesa = self.mesa
        # ¿Tengo alguna carta del mismo color?
        tiene_color = any(c == color_mesa for c, n in self.mano_agente)
        # ¿Tengo alguna carta del mismo número?
        tiene_num = any(n == num_mesa for c, n in self.mano_agente)

        # El estado es mucho más simple: (ColorMesa, NumMesa, TengoColor?, TengoNum?, CuantasCartas)
        return (color_mesa, num_mesa, tiene_color, tiene_num, len(self.mano_agente))

    def reset(self, seed=None):
        self.mazo = self.deck_base.copy()
        random.shuffle(self.mazo)
        self.mano_agente = [self.mazo.pop() for _ in range(5)]
        self.mano_rival = [self.mazo.pop() for _ in range(5)]
        self.mesa = self.mazo.pop()
        return self._get_state(), {}

    def step(self, action):
        reward = 0
        done = False
        color_m, num_m = self.mesa

        if action == 0:  # Intentar jugar por color
            cartas_validas = [c for c in self.mano_agente if c[0] == color_m]
            if cartas_validas:
                carta = cartas_validas[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 10  # Gran premio por jugar legal
            else:
                reward = -5  # Castigo por intentar jugar color sin tener
                action = 2  # Si falla, le obligamos a robar

        elif action == 1:  # Intentar jugar por número
            cartas_validas = [c for c in self.mano_agente if c[1] == num_m]
            if cartas_validas:
                carta = cartas_validas[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 10
            else:
                reward = -5
                action = 2

        if action == 2:  # Robar
            if self.mazo:
                self.mano_agente.append(self.mazo.pop())
            reward = -1  # Pequeño castigo por no jugar

        # Condición de victoria
        if len(self.mano_agente) == 0:
            return self._get_state(), 100, True, False, {"msg": "GANASTE"}

        # Turno del rival (Bot tonto)
        for c in self.mano_rival:
            if c[0] == self.mesa[0] or c[1] == self.mesa[1]:
                self.mano_rival.remove(c)
                self.mesa = c
                break
        else:
            if self.mazo:
                self.mano_rival.append(self.mazo.pop())

        if len(self.mano_rival) == 0:
            return self._get_state(), -50, True, False, {"msg": "PERDISTE"}

        if not self.mazo:
            done = True
        return self._get_state(), reward, done, False, {}


# --- Entrenamiento ---
env = UnoSimplificado()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
episodes = 50000
alpha, gamma = 0.2, 0.9
epsilon, decay = 1.0, 0.9999

for i in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() if random.random(
        ) < epsilon else np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        q_table[state][action] += alpha * \
            (reward + gamma *
             np.max(q_table[next_state]) - q_table[state][action])
        state = next_state
    epsilon = max(0.01, epsilon * decay)

# --- Prueba ---
state, _ = env.reset()
done = False
print("\n--- PARTIDA CON ESTADOS AGRUPADOS ---")
while not done:
    color_m, num_m, t_c, t_n, n_c = state
    print(
        f"Mesa: {color_m}{num_m} | Cartas en mano: {n_c} | ¿Tengo color?: {t_c}")
    action = np.argmax(q_table[state])
    action_name = ["Jugar Color", "Jugar Numero", "Robar"][action]
    print(f"Agente decide: {action_name}")
    state, reward, done, _, info = env.step(action)
    if "msg" in info:
        print(f"RESULTADO: {info['msg']}")
