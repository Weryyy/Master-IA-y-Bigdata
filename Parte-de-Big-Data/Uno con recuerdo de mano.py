import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict


class UnoEstrategico(gym.Env):
    def __init__(self):
        super(UnoEstrategico, self).__init__()
        self.colores = ['R', 'A', 'V']
        self.numeros = [1, 2, 3]
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]

        # ACCIONES:
        # 0: Jugar la carta legal MÁS BAJA
        # 1: Jugar la carta legal MÁS ALTA
        # 2: Robar
        self.action_space = spaces.Discrete(3)

    def _get_state(self):
        color_m, num_m = self.mesa

        # Filtramos qué cartas podría tirar según la regla (Color o Num >= Mesa)
        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] >= num_m]

        # "Memoria" simplificada de la mano:
        # Guardamos si tenemos cartas de valor 1, 2 o 3 en las legales
        tiene_1 = any(c[1] == 1 for c in legales)
        tiene_2 = any(c[1] == 2 for c in legales)
        tiene_3 = any(c[1] == 3 for c in legales)

        # El estado combina: Mesa, y qué tipo de opciones legales tengo en mano
        return (color_m, num_m, tiene_1, tiene_2, tiene_3, len(self.mano_agente))

    def step(self, action):
        reward = 0
        done = False
        color_m, num_m = self.mesa

        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] >= num_m]

        if action == 0:  # JUGAR MÁS BAJA
            if legales:
                legales.sort(key=lambda x: x[1])  # Orden ascendente
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 10
            else:
                reward = -15  # Castigo por intentar jugar sin tener
                action = 2  # Forzar robo

        elif action == 1:  # JUGAR MÁS ALTA
            if legales:
                # Orden descendente
                legales.sort(key=lambda x: x[1], reverse=True)
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                # Un poco más de recompensa por jugar cartas altas (limpiar mesa)
                reward = 12
            else:
                reward = -15
                action = 2

        if action == 2:  # ROBAR
            if self.mazo:
                self.mano_agente.append(self.mazo.pop())
            reward = -2

        # Victoria
        if len(self.mano_agente) == 0:
            return self._get_state(), 100, True, False, {"msg": "GANASTE"}

        # Rival (Juega aleatorio legal)
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

        # Si se acaba el mazo, rebarajamos el descarte (simplificado)
        if not self.mazo:
            self.mazo = self.deck_base.copy()
            random.shuffle(self.mazo)

        return self._get_state(), reward, done, False, {}

    def reset(self, seed=None):
        self.mazo = self.deck_base.copy() * 3  # Mazo grande para partidas largas
        random.shuffle(self.mazo)
        self.mano_agente = [self.mazo.pop() for _ in range(5)]
        self.mano_rival = [self.mazo.pop() for _ in range(5)]
        self.mesa = self.mazo.pop()
        return self._get_state(), {}


# --- Entrenamiento ---
env = UnoEstrategico()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
episodes = 60000
alpha, gamma, epsilon = 0.1, 0.95, 1.0

print("Entrenando estrategia de selección...")
for i in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        q_table[state][action] += alpha * \
            (reward + gamma *
             np.max(q_table[next_state]) - q_table[state][action])
        state = next_state
    epsilon = max(0.01, epsilon * 0.99992)

# --- Prueba de la IA ---
state, _ = env.reset()
done = False
print("\n--- IA DECIDIENDO QUÉ CARTA SOLTAR ---")
while not done:
    c_m, n_m, t1, t2, t3, n_c = state
    print(f"Mesa: {c_m}{n_m} | Cartas: {n_c} | Opciones: 1:{t1}, 2:{t2}, 3:{t3}")

    action = np.argmax(q_table[state])
    nombres = ["TIRAR BAJA", "TIRAR ALTA", "ROBAR"]
    print(f"IA elige: {nombres[action]}")

    state, reward, done, _, info = env.step(action)
    if "msg" in info:
        print(f"RESULTADO FINAL: {info['msg']}")
