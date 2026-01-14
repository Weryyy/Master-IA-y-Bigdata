import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict


class UnoConJerarquia(gym.Env):
    def __init__(self):
        super(UnoConJerarquia, self).__init__()
        self.colores = ['R', 'A', 'V']  # 3 Colores
        self.numeros = [1, 2, 3]       # Números del 1 al 3
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]
        # Acciones: 0: Jugar carta legal, 1: Robar
        self.action_space = spaces.Discrete(2)

    def _get_state(self):
        color_m, num_m = self.mesa
        # Buscamos si tenemos jugadas legales bajo la nueva regla
        tiene_jugada = any(c == color_m or n >= num_m for c,
                           n in self.mano_agente)
        # ¿Tengo alguna carta de mayor valor exacto? (Para que aprenda a reservar)
        tiene_mayor = any(n > num_m for c, n in self.mano_agente)

        return (color_m, num_m, tiene_jugada, tiene_mayor, len(self.mano_agente))

    def step(self, action):
        reward = 0
        done = False
        color_m, num_m = self.mesa

        if action == 0:  # Intentar Jugar
            # Filtrar cartas legales: mismo color O número >= mesa
            legales = [c for c in self.mano_agente if c[0]
                       == color_m or c[1] >= num_m]

            if legales:
                # El agente juega la carta más alta que tiene de las legales (estrategia simple)
                legales.sort(key=lambda x: x[1], reverse=True)
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 15  # Premio por jugada legal exitosa
            else:
                reward = -20  # Castigo por intentar jugar sin tener cartas legales
                action = 1  # Le obligamos a robar

        if action == 1:  # Robar
            if self.mazo:
                self.mano_agente.append(self.mazo.pop())
            reward = -2  # Castigo por robar (queremos que juegue si puede)

        # --- Lógica de victoria y rival ---
        if len(self.mano_agente) == 0:
            return self._get_state(), 100, True, False, {"msg": "GANASTE"}

        # Rival (usa la misma regla)
        legales_rival = [c for c in self.mano_rival if c[0]
                         == self.mesa[0] or c[1] >= self.mesa[1]]
        if legales_rival:
            c = random.choice(legales_rival)
            self.mano_rival.remove(c)
            self.mesa = c
        elif self.mazo:
            self.mano_rival.append(self.mazo.pop())

        if len(self.mano_rival) == 0:
            return self._get_state(), -50, True, False, {"msg": "PERDISTE"}
        if not self.mazo:
            done = True
        return self._get_state(), reward, done, False, {}

    def reset(self, seed=None):
        self.mazo = self.deck_base.copy() * 2  # Mazo doble para que no se acabe
        random.shuffle(self.mazo)
        self.mano_agente = [self.mazo.pop() for _ in range(4)]
        self.mano_rival = [self.mazo.pop() for _ in range(4)]
        self.mesa = self.mazo.pop()
        return self._get_state(), {}


# --- Entrenamiento ---
env = UnoConJerarquia()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
episodes = 30000
alpha, gamma, epsilon = 0.2, 0.9, 1.0

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
    epsilon = max(0.01, epsilon * 0.9999)

# --- Prueba Visual ---
state, _ = env.reset()
done = False
print("\n--- TEST: REGLA DE JERARQUÍA (COLOR O NÚMERO >=) ---")
while not done:
    c_m, n_m, tiene_j, tiene_may, n_c = state
    print(f"Mesa: {c_m}{n_m} | Cartas: {n_c} | ¿Puedo jugar?: {tiene_j}")
    action = np.argmax(q_table[state])
    print(f"Agente decide: {'JUGAR' if action == 0 else 'ROBAR'}")
    state, reward, done, _, info = env.step(action)
    if "msg" in info:
        print(f"RESULTADO: {info['msg']}")
