import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict
import time


class UnoVisual(gym.Env):
    def __init__(self):
        super(UnoVisual, self).__init__()
        self.colores = ['Rojo', 'Azul', 'Verde']
        self.numeros = [1, 2, 3]
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]
        self.action_space = spaces.Discrete(3)  # 0:Baja, 1:Alta, 2:Robar

    def _get_state(self):
        color_m, num_m = self.mesa
        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] >= num_m]
        t1 = any(c[1] == 1 for c in legales)
        t2 = any(c[1] == 2 for c in legales)
        t3 = any(c[1] == 3 for c in legales)
        return (color_m, num_m, t1, t2, t3, len(self.mano_agente))

    def reset(self, seed=None):
        self.mazo = self.deck_base.copy() * 3
        random.shuffle(self.mazo)
        self.mano_agente = [self.mazo.pop() for _ in range(5)]
        self.mano_rival = [self.mazo.pop() for _ in range(5)]
        self.mesa = self.mazo.pop()
        return self._get_state(), {}

    def step(self, action):
        reward = 0
        done = False
        color_m, num_m = self.mesa
        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] >= num_m]

        if action == 0:  # JUGAR MÁS BAJA
            if legales:
                legales.sort(key=lambda x: x[1])
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 10
            else:
                reward = -20  # Castigo por intentar jugar sin poder

        elif action == 1:  # JUGAR MÁS ALTA
            if legales:
                legales.sort(key=lambda x: x[1], reverse=True)
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 12  # Premio mayor por usar cartas altas (estrategia)
            else:
                reward = -20

        elif action == 2:  # ROBAR
            if self.mazo:
                self.mano_agente.append(self.mazo.pop())
            reward = -2  # Pequeño castigo por no avanzar

        # Victoria/Derrota
        if len(self.mano_agente) == 0:
            return self._get_state(), 100, True, False, {"msg": "GANASTE"}

        # Turno Rival
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
        return self._get_state(), reward, False, False, {}


# --- Entrenamiento Rápido ---
env = UnoVisual()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
for i in range(40000):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() if random.random(
        ) < 0.1 else np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        q_table[state][action] += 0.1 * \
            (reward + 0.9 *
             np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

# --- DEMOSTRACIÓN CON LOGS DE REWARDS ---
print("\n" + "="*50)
print("SISTEMA DE RECOMPENSAS Y ANÁLISIS DE CARTAS")
print("="*50)

state, _ = env.reset()
done = False
while not done:
    color_m, num_m = env.mesa
    legales = [c for c in env.mano_agente if c[0] == color_m or c[1] >= num_m]

    print(f"\n[ESTADO] Mesa: {color_m} {num_m}")
    print(f"[MANO AGENTE]: {env.mano_agente}")
    print(
        f"[OPCIONES LEGALES]: {legales if legales else 'Ninguna (Debe robar)'}")

    action = np.argmax(q_table[state])
    action_name = ["JUGAR BAJA", "JUGAR ALTA", "ROBAR"][action]

    # Predecir qué pasaría (Simulamos el step para el print)
    next_state, reward, done, _, info = env.step(action)

    print(f"[IA ELIGE]: {action_name}")
    print(f"[RECOMPENSA]: {reward} puntos")

    if reward > 0:
        print("✅ Acción válida: Sumando puntos al conocimiento.")
    elif reward == -2:
        print("⚠️ Forzado a robar: Penalización leve.")
    else:
        print("❌ Error táctico: Castigo por movimiento ilegal.")

    state = next_state
    if "msg" in info:
        print(f"\n--- {info['msg']} ---")
    time.sleep(1)
