#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>

// Структура для возврата результата из итерационных методов
struct IterationResult {
    std::vector<float> solution;
    int iterations;
};

// Вспомогательная функция для печати вектора
void printVector(const std::vector<float>& vec, const std::string& name, int precision = 4) {
    std::cout << name << ":" << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(precision) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// Вспомогательная функция для печати матрицы
void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << val;
        }
        std::cout << std::endl;
    }
}

// Метод Якоби
IterationResult jacobi_method(const std::vector<std::vector<float>>& A, const std::vector<float>& f, float epsilon, int k_max) {
    int n = f.size();
    std::vector<float> x(n, 0.0f); // Начальное приближение - нулевой вектор
    std::vector<float> x_new(n, 0.0f);

    for (int k = 0; k < k_max; ++k) {
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (f[i] - sum) / A[i][i];
        }

        // Проверка критерия остановки
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
        }

        x = x_new;

        if (max_diff < epsilon) {
            return {x, k + 1};
        }
    }
    return {x, k_max}; // Выход по максимальному числу итераций
}

// Метод последовательной верхней релаксации (SOR)
IterationResult sor_method(const std::vector<std::vector<float>>& A, const std::vector<float>& f, float omega, float epsilon, int k_max) {
    int n = f.size();
    std::vector<float> x(n, 0.0f); // Начальное приближение

    for (int k = 0; k < k_max; ++k) {
        std::vector<float> x_old = x;
        for (int i = 0; i < n; ++i) {
            float sum1 = 0.0f;
            for (int j = 0; j < i; ++j) {
                sum1 += A[i][j] * x[j]; // Используются уже обновленные компоненты x
            }
            float sum2 = 0.0f;
            for (int j = i + 1; j < n; ++j) {
                sum2 += A[i][j] * x_old[j]; // Используются компоненты с предыдущей итерации
            }
            
            float new_val = (f[i] - sum1 - sum2) / A[i][i];
            x[i] = (1 - omega) * x_old[i] + omega * new_val;
        }

        // Проверка критерия остановки
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(x[i] - x_old[i]));
        }

        if (max_diff < epsilon) {
            return {x, k + 1};
        }
    }
    return {x, k_max};
}

int main() {
    // --- 1. Определение параметров ---
    const int n = 10;
    const int m = 15;
    const float epsilon = 0.0001f;
    const int k_max = 1000;

    std::cout << "========================================================" << std::endl;
    std::cout << " Входные данные" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Параметры: n = " << n << ", m = " << m << ", epsilon = " << epsilon << ", k_max = " << k_max << std::endl << std::endl;

    // --- 2. Генерация данных ---
    std::vector<std::vector<float>> A(n, std::vector<float>(n));
    std::vector<float> x_exact(n);
    std::vector<float> f(n, 0.0f);

    srand(time(0)); // Инициализация генератора случайных чисел

    // Генерация недиагональных элементов и вычисление диагональных
    for (int i = 0; i < n; ++i) {
        float row_sum_abs = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A[i][j] = rand() % 5 - 4; // Случайное число от -4 до 0 [cite: 301]
                row_sum_abs += std::abs(A[i][j]);
            }
        }
        A[i][i] = row_sum_abs;
    }
    // Обеспечение строгого диагонального преобладания для первой строки
    A[0][0] += 1.0f; 

    // Генерация точного вектора решений x
    for (int i = 0; i < n; ++i) {
        x_exact[i] = m + i;
    }

    // Вычисление вектора правой части f = A * x
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i] += A[i][j] * x_exact[j];
        }
    }

    printMatrix(A, "Сгенерированная матрица A");
    printVector(x_exact, "\nТочный вектор решений x_exact", 0);
    printVector(f, "\nВычисленный вектор правой части f", 2);
    
    std::cout << "\n========================================================" << std::endl;
    std::cout << " Выходные данные" << std::endl;
    std::cout << "========================================================" << std::endl;

    // --- 3. Выполнение и вывод результатов ---
    
    // Метод Якоби
    auto result_jacobi = jacobi_method(A, f, epsilon, k_max);
    std::cout << "\n--- Метод Якоби ---" << std::endl;
    std::cout << "Количество итераций: " << result_jacobi.iterations << std::endl;
    printVector(result_jacobi.solution, "Приближенный вектор решений");

    // Метод релаксации (ω = 0.5)
    auto result_sor05 = sor_method(A, f, 0.5f, epsilon, k_max);
    std::cout << "\n--- Метод Релаксации (w = 0.5) ---" << std::endl;
    std::cout << "Количество итераций: " << result_sor05.iterations << std::endl;
    printVector(result_sor05.solution, "Приближенный вектор решений");

    // Метод Гаусса-Зейделя (ω = 1.0)
    auto result_gs = sor_method(A, f, 1.0f, epsilon, k_max);
    std::cout << "\n--- Метод Гаусса-Зейделя (w = 1.0) ---" << std::endl;
    std::cout << "Количество итераций: " << result_gs.iterations << std::endl;
    printVector(result_gs.solution, "Приближенный вектор решений");

    // Метод релаксации (ω = 1.5)
    auto result_sor15 = sor_method(A, f, 1.5f, epsilon, k_max);
    std::cout << "\n--- Метод Релаксации (w = 1.5) ---" << std::endl;
    std::cout << "Количество итераций: " << result_sor15.iterations << std::endl;
    printVector(result_sor15.solution, "Приближенный вектор решений");
    
    if(result_sor05.iterations >= k_max){
        std::cout << "\nСообщение: Метод релаксации (w = 0.5) вышел из-за превышения допустимого количества итераций." << std::endl; 
    }

    return 0;
}