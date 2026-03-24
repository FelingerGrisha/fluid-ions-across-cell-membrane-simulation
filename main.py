import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Импортируем параметры и начальные значения
from init_erith0 import *
import init_pump
from pump import pump

# Параметры воды
Pw: float = 1.5e-2  # Проницаемость мембраны для воды, см/с
Vw: float = 18.0    # Молярный объем воды, см^(3)/моль

# Начальные параметры проницаемостей
PNa0: float = PNa
PK0: float = PK
PCa0: float = PCa

# Параметры расчета
dt: float = 1e-3           # Шаг времени
Nmax: int = 30 * 60000  # Максимальное число итераций
ndisp: int = 1000          # Частота вывода данных
th: float = 20.0           # Временная константа
tmm: float = 5.0           # Время активации

# Создание папки для сохранения графиков
plots_dir: str = 'plots'
# Файл для сохранения промежуточного состояния (intermediateState)
Intermediate_state: str = 'intermediateState.pkl'


def save_IntermediateState(i: int, t: float, n_Na: float, n_K: float, n_Cl: float,
                           n_Ca: float, Vc: float, Em: float, tt: list, nn_Na: list,
                           nn_K: list, nn_Cl: list, nn_Ca: list, Vcc: list, Emm: list) -> None:
    """Функция для сохранения промежуточного состояния"""
    state = {
        'i': i, 't': t, 'n_Na': n_Na, 'n_K': n_K, 'n_Cl': n_Cl,
        'n_Ca': n_Ca, 'Vc': Vc, 'Em': Em, 'tt': tt, 'nn_Na': nn_Na,
        'nn_K': nn_K, 'nn_Cl': nn_Cl, 'nn_Ca': nn_Ca, 'Vcc': Vcc, 'Emm': Emm
    }
    with open(Intermediate_state, 'wb') as cp_file:
        pickle.dump(state, cp_file)
    print(f"\nПромежуточное состояние сохранено в {Intermediate_state} на итерации {i} (t = {t:.4f})")


def get_derivatives(t_curr: float, Y: np.ndarray) -> tuple[np.ndarray, float, tuple]:
    """Функция для вычисления правых частей дифференциальных уравнений (скоростей)"""
    # Распаковка вектора текущего состояния
    curr_n_Na, curr_n_K, curr_n_Cl, curr_n_Ca, curr_Vc = Y

    # Расчет текущего мембранного потенциала
    curr_Em = F * (curr_n_Na + curr_n_K + 2 * curr_n_Ca + 2 * n_Mg - curr_n_Cl + zx * n_Xi) / (Ac * Cm)
    u_curr = F * curr_Em / (R * T)

    # Расчет эпсилон-функций для потоков
    if abs(u_curr) >= 0.00001:
        epsm = u_curr / (np.exp(0.5 * u_curr) - np.exp(-0.5 * u_curr))
        epsm2 = 2 * u_curr / (np.exp(u_curr) - np.exp(-u_curr))
    else:
        epsm = 1.0
        epsm2 = 1.0

    # Активация/деактивация транспорта
    cur_PNa = 4 * PNa0 if t_curr >= tmm else PNa0
    cur_PK = 4 * PK0 if t_curr >= tmm else PK0
    cur_PCa = 4 * PCa0 if t_curr >= tmm else PCa0

    # Осмотические концентрации
    Pii = (curr_n_Na + curr_n_K + curr_n_Cl + curr_n_Ca + n_Xi) / curr_Vc

    # Проницаемость Ca^(2+)-зависимых K^(+)-каналов
    Pch = PKmax * (curr_n_Ca / (curr_n_Ca + curr_Vc * Kch)) ** Nch

    # Электродиффузионные потоки, моль/(см^2*с)
    JNa = cur_PNa * epsm * (Na_e * np.exp(-0.5 * u_curr) - curr_n_Na * np.exp(0.5 * u_curr) / curr_Vc)
    JK = (cur_PK + Pch) * epsm * (K_e * np.exp(-0.5 * u_curr) - curr_n_K * np.exp(0.5 * u_curr) / curr_Vc)
    JCl = PCl * epsm * (Cl_e * np.exp(0.5 * u_curr) - curr_n_Cl * np.exp(-0.5 * u_curr) / curr_Vc)
    JCa = cur_PCa * epsm2 * (Ca_e * np.exp(-u_curr) - curr_n_Ca * np.exp(u_curr) / curr_Vc)

    # Поток Ca^(2+)-АТФазы
    J_CaATP = Q_CaATP * (curr_n_Ca / (curr_n_Ca + curr_Vc * K_CaATP)) ** 2

    # Na^(+)/K^(+)-АТФаза
    Na_i = curr_n_Na / curr_Vc
    K_i = curr_n_K / curr_Vc
    Ap = pump(u_curr, Na_i, Na_e, K_i, K_e,
              init_pump.k12, init_pump.k23, init_pump.k34o, init_pump.k45,
              init_pump.k56, init_pump.k61, init_pump.k21, init_pump.k32,
              init_pump.k43o, init_pump.k54, init_pump.k65, init_pump.k16,
              init_pump.ATP, init_pump.ADP, init_pump.P_i)
    Jp = N * Ap

    # Скорости изменения количества ионов и объема
    d_Na_dt = Ac * (-3 * Jp + JNa)
    d_K_dt = Ac * (2 * Jp + JK)
    d_Cl_dt = Ac * JCl
    d_Ca_dt = Ac * (-2 * J_CaATP + JCa)
    dVc_dt = Ac * Vw * Pw * ((curr_n_Na + curr_n_K + curr_n_Cl + curr_n_Ca + n_Mg + n_Xi) / curr_Vc - Pie)

    derivatives = np.array([d_Na_dt, d_K_dt, d_Cl_dt, d_Ca_dt, dVc_dt])
    fluxes = (JNa, JK, JCl, JCa, Jp, J_CaATP)

    return derivatives, curr_Em, fluxes


def main() -> None:
    # Объявляем глобальными переменные, которые импортированы из init_erith0 и будут перезаписываться
    global n_Na, n_K, n_Cl, n_Ca, Vc

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    t: float = 0.0
    # Начальный мембранный потенциал, В
    Em: float = F * (n_Na + n_K + 2 * n_Ca + 2 * n_Mg - n_Cl + zx * n_Xi) / (Ac * Cm)

    # Инициализация массивов для хранения данных
    tt: list[float] = [t]
    nn_Na: list[float] = [n_Na]
    nn_K: list[float] = [n_K]
    nn_Cl: list[float] = [n_Cl]
    nn_Ca: list[float] = [n_Ca]
    Vcc: list[float] = [Vc]
    Emm: list[float] = [Em]

    start_i: int = 0
    file_mode: str = 'w'

    # Проверка и загрузка промежуточного состояния при запуске
    if os.path.exists(Intermediate_state):
        with open(Intermediate_state, 'rb') as cp_file:
            state = pickle.load(cp_file)

        start_i = state['i'] + 1
        t = state['t']
        n_Na = state['n_Na']
        n_K = state['n_K']
        n_Cl = state['n_Cl']
        n_Ca = state['n_Ca']
        Vc = state['Vc']
        Em = state['Em']
        tt = state['tt']
        nn_Na = state['nn_Na']
        nn_K = state['nn_K']
        nn_Cl = state['nn_Cl']
        nn_Ca = state['nn_Ca']
        Vcc = state['Vcc']
        Emm = state['Emm']
        file_mode = 'a'
        print(f"Возобновление с итерации {start_i} (t = {t:.4f})")
    else:
        print("Новый запуск программы")

    i: int = start_i # Инициализируем i до try на случай прерывания до начала цикла

    try:
        # Открытие файлов для записи данных
        with open('conc.dat', file_mode) as ff, \
             open('flux.dat', file_mode) as ff1, \
             open('ion.dat', file_mode) as ff2:

            if file_mode == 'w':
                ff.write(f"{t} {n_Na} {n_K} {n_Cl} {n_Ca} {Em} {Vc}\n")

            for i in range(start_i, Nmax):
                # Вектор текущего состояния перед шагом
                Y = np.array([n_Na, n_K, n_Cl, n_Ca, Vc])

                # сам метод Рунге-Кутты 4-го порядка
                k1, Em_k1, fluxes_k1 = get_derivatives(t, Y)
                k2, Em_k2, fluxes_k2 = get_derivatives(t + 0.5 * dt, Y + 0.5 * dt * k1)
                k3, Em_k3, fluxes_k3 = get_derivatives(t + 0.5 * dt, Y + 0.5 * dt * k2)
                k4, Em_k4, fluxes_k4 = get_derivatives(t + dt, Y + dt * k3)

                # Вычисление итогового приращения состояния за время dt
                delta_Y = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                # Перерасчёт количества ионов и объема
                Y = Y + delta_Y
                n_Na, n_K, n_Cl, n_Ca, Vc = Y

                t += dt

                # Периодический вывод данных в файлы
                if (i % ndisp) == 0:
                    # Потоки для логов берем из первого (реального) шага k1
                    JNa, JK, JCl, JCa, Jp, J_CaATP = fluxes_k1

                    # Приращения ионов для файла ion.dat
                    dn_Na = delta_Y[0]
                    dn_K = delta_Y[1]
                    dn_Cl = delta_Y[2]
                    dn_Ca = delta_Y[3]

                    # Актуальный мембранный потенциал на данный момент времени
                    Em = F * (n_Na + n_K + 2 * n_Ca + 2 * n_Mg - n_Cl + zx * n_Xi) / (Ac * Cm)

                    ff.write(f"{t} {n_Na} {n_K} {n_Cl} {n_Ca} {Em} {Vc}\n")
                    ff1.write(f"{t} {JNa} {JK} {JCl} {JCa} {Jp} {J_CaATP}\n")
                    ff2.write(f"{t} {dn_Na} {dn_K} {dn_Cl} {dn_Ca}\n")

                    tt.append(t)
                    nn_Na.append(n_Na)
                    nn_K.append(n_K)
                    nn_Cl.append(n_Cl)
                    nn_Ca.append(n_Ca)
                    Vcc.append(Vc)
                    Emm.append(Em)

    except KeyboardInterrupt:
        # Обработка прерывания
        Em = F * (n_Na + n_K + 2 * n_Ca + 2 * n_Mg - n_Cl + zx * n_Xi) / (Ac * Cm)
        save_IntermediateState(i, t, n_Na, n_K, n_Cl, n_Ca, Vc, Em, tt, nn_Na, nn_K, nn_Cl, nn_Ca, Vcc, Emm)
        print("\nПрограмма прервана. Состояние сохранено.")
        sys.exit(0)

    # Удаляем intermediateState при успешном завершении
    if os.path.exists(Intermediate_state):
        os.remove(Intermediate_state)
        print(f"\nПрограмма завершена успешно. Файл {Intermediate_state} удалён.")

    # Вывод итоговых значений в консоль
    print("\n" + "="*60)
    print("Итоговые результаты моделирования")
    print("="*60)

    _, Em_final, final_fluxes = get_derivatives(t, np.array([n_Na, n_K, n_Cl, n_Ca, Vc]))
    JNa_f, JK_f, JCl_f, JCa_f, Jp_f, J_CaATP_f = final_fluxes

    print(f"Cl_i = {Cl_i}")
    print(f"Xi_i = {Xi_i}")
    print(f"zx = {zx}")
    print(f"Q_CaATP = {Q_CaATP}")

    print("\nКоличества ионов (моль):")
    print(f"  Na⁺: {n_Na:.2e} моль")
    print(f"  K⁺: {n_K:.2e} моль")
    print(f"  Cl⁻: {n_Cl:.2e} моль")
    print(f"  Ca²⁺: {n_Ca:.2e} моль")

    print("\nФизиологические параметры:")
    print(f"  Мембранный потенциал (E_m): {Em_final*1000:.2f} мВ")
    print(f"  Объем клетки (V_c): {Vc*1e12:.2f} мкм³")

    print("\nПотоки ионов (моль/(см^2*с)):")
    print(f"  J_Na: {JNa_f:.2e} моль/(см^2*с)")
    print(f"  J_K: {JK_f:.2e} моль/(см^2*с)")
    print(f"  J_Cl: {JCl_f:.2e} моль/(см^2*с)")
    print(f"  J_Ca: {JCa_f:.2e} моль/(см^2*с)")
    print(f"  J_p: {Jp_f:.2e} моль/(см^2*с)")
    print(f"  J_CaATP: {J_CaATP_f:.2e} моль/(см^2*с)")

    # Построение и сохранение графиков
    plt.figure()
    plt.plot(tt, nn_Na, label='n_Na')
    plt.plot(tt, nn_K, label='n_K')
    plt.plot(tt, nn_Cl, label='n_Cl')
    plt.plot(tt, nn_Ca, label='n_Ca')
    plt.legend()
    plt.xlabel('Время t (сек)')
    plt.ylabel('Количества ионов n (моль)')
    plt.title('Зависимость количества ионов n от времени t')
    plt.savefig(os.path.join(plots_dir, 'ion_amounts.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)

    plt.figure(figsize=(8, 5))
    plt.plot(tt, Emm, label='Em')
    plt.legend()
    plt.xlabel('Время t (сек)')
    plt.ylabel('Мембранный потенциал Em (В)')
    plt.title('Зависимость мембранного потенциала Em от времени t')
    plt.savefig(os.path.join(plots_dir, 'membrane_potential.png'), dpi=500, bbox_inches='tight')
    plt.show(block=False)

    plt.figure()
    plt.plot(tt, Vcc, label='Vc')
    plt.legend()
    plt.xlabel('Время t (сек)')
    plt.ylabel('Клеточный объём Vc (см³)')
    plt.title('Зависимоcть клеточного объёма Vc от времени t')
    plt.savefig(os.path.join(plots_dir, 'cell_volume.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)

if __name__ == "__main__":
    main()