#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

// Функция для вычисления L2-нормы вектора
float calculateL2Norm(const std::vector<float>& vec) {
    float sum_of_squares = 0.0f;
    for (float val : vec) {
        sum_of_squares += val * val;
    }
    return std::sqrt(sum_of_squares);
}

// Функция для вычисления чебышевской нормы (C-нормы) вектора
float calculateCNorm(const std::vector<float>& vec) {
    float max_abs_val = 0.0f;
    for (float val : vec) {
        if (std::abs(val) > max_abs_val) {
            max_abs_val = std::abs(val);
        }
    }
    return max_abs_val;
}


int main() {
    // --- 1. Входные данные ---
    const int N_plus_1 = 2000; // Порядок матрицы
    const int N = N_plus_1 - 1;
    const float m = 15.0f;      
    const float k = 4.0f;       

    
    setlocale(LC_ALL, "Russian");

    // --- 2. Формирование системы Ax = f ---

    // Векторы для хранения диагоналей матрицы и правой части
    std::vector<float> a(N_plus_1), b(N_plus_1), c(N_plus_1), f(N_plus_1);

    // Задаем вектор точного решения y, как указано в задании
    std::vector<float> y_exact(N_plus_1);
    for (int i = 0; i < N_plus_1; ++i) {
        y_exact[i] = static_cast<float>(i + 1);
    }

    // Формируем коэффициенты трех диагоналей матрицы A
    // a - нижняя диагональ, c - главная, b - верхняя
    for(int i = 0; i < N_plus_1; ++i) {
        a[i] = k;
        b[i] = k;
    }
    c[0] = m;
    for (int i = 1; i < N_plus_1; ++i) {
        c[i] = m + k + static_cast<float>(i);
    }

    // Вычисляем вектор правой части f = A * y_exact
    // Первое уравнение: c_0*y_0 - b_0*y_1 = f_0
    f[0] = c[0] * y_exact[0] - b[0] * y_exact[1];

    // Уравнения с i = 1 по N-1: -a_i*y_{i-1} + c_i*y_i - b_i*y_{i+1} = f_i
    for (int i = 1; i < N; ++i) {
        f[i] = -a[i] * y_exact[i-1] + c[i] * y_exact[i] - b[i] * y_exact[i+1];
    }

    // Последнее уравнение: -a_N*y_{N-1} + c_N*y_N = f_N
    f[N] = -a[N] * y_exact[N-1] + c[N] * y_exact[N];

    // --- 3. Реализация метода правой прогонки ---

    // Засекаем время начала вычислений
    auto start = std::chrono::high_resolution_clock::now();

    // Прямая прогонка: вычисление коэффициентов alpha и beta
    std::vector<float> alpha(N + 2), beta(N + 2);

    alpha[1] = b[0] / c[0];
    beta[1] = f[0] / c[0];

    for (int i = 1; i < N; ++i) {
        float denominator = c[i] - a[i] * alpha[i];
        alpha[i+1] = b[i] / denominator;
        beta[i+1] = (f[i] + a[i] * beta[i]) / denominator;
    }

    float denominator_N = c[N] - a[N] * alpha[N];
    beta[N+1] = (f[N] + a[N] * beta[N]) / denominator_N;

    // Обратная прогонка: вычисление вектора решения y*
    std::vector<float> y_star(N_plus_1);
    y_star[N] = beta[N+1];

    for (int i = N - 1; i >= 0; --i) {
        y_star[i] = alpha[i+1] * y_star[i+1] + beta[i+1];
    }

    // Засекаем время окончания вычислений
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // --- 4. Расчет погрешности ---

    // Вектор разницы между точным и приближенным решениями
    std::vector<float> diff(N_plus_1);
    for (int i = 0; i < N_plus_1; ++i) {
        diff[i] = y_exact[i] - y_star[i];
    }

    // Относительная погрешность
    float relative_error_l2 = calculateL2Norm(diff) / calculateL2Norm(y_exact);
    float relative_error_c = calculateCNorm(diff) / calculateCNorm(y_exact);

    // --- 5. Представление результатов ---

    std::cout << "--- Результаты вычислений ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6); // Устанавливаем формат вывода для float

    std::cout << "\n1. Первые 5 координат вектора приближённого решения y*:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "y*[" << i << "] = " << y_star[i] << std::endl;
    }

    std::cout << "\n2. Относительная погрешность:" << std::endl;
    std::cout << std::scientific; // Переключаемся на научную нотацию для погрешности
    std::cout << "   - По норме L2: " << relative_error_l2 << std::endl;
    std::cout << "   - По норме C (чебышевской): " << relative_error_c << std::endl;

    std::cout << "\n3. Время выполнения: " << elapsed.count() << " секунд" << std::endl;

    return 0;
}