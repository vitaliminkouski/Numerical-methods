#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>

// Используем псевдонимы для удобства
using matrix = std::vector<std::vector<double>>;
using vec = std::vector<double>;


//Функция для вывода первых 'count' элементов вектора.
void print_vector(const vec& v, const std::string& title, int count) {
    std::cout << title << std::endl;
    if (v.empty()) {
        std::cout << "  (решение не найдено)" << std::endl;
        return;
    }
    for (int i = 0; i < count && i < v.size(); ++i) {
        std::cout << "  x*[" << i << "] = " << v[i] << std::endl;
    }
}

/**
 * @brief Функция для вычисления евклидовой нормы вектора.
 */
double vector_norm(const vec& v) {
    double norm = 0.0;
    for (double val : v) {
        norm += val * val;
    }
    return std::sqrt(norm);
}

/**
 * @brief Вычисляет вектор разности v1 - v2.
 */
vec vector_diff(const vec& v1, const vec& v2) {
    if (v1.size() != v2.size()) return {};
    vec diff(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        diff[i] = v1[i] - v2[i];
    }
    return diff;
}


//Вычисляет норму вектора невязки ||Ax - b||
double calculate_residual_norm(const matrix& A, const vec& x, const vec& b) {
    int n = A.size();
    if (x.empty()) return -1.0; 
    
    vec Ax(n, 0.0);
    for(int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Ax[i] += A[i][j] * x[j];
        }
    }
    
    vec residual = vector_diff(Ax, b);
    
    return vector_norm(residual);
}


// Решение СЛАУ на основе LDLT-разложения для симметричных матриц.
vec solve_ldlt(matrix A, vec b) {
    int n = A.size();
    vec t(n);

    for (int k = 0; k < n - 1; ++k) {
        double d_kk = A[k][k];
        if (std::abs(d_kk) < 1e-12) {
             std::cerr << "Ошибка LDLT: Нулевой диагональный элемент." << std::endl;
             return {};
        }
        for (int i = k + 1; i < n; ++i) {
            t[i] = A[i][k];
            A[i][k] /= d_kk;
        }
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j <= i; ++j) {
                 A[i][j] -= A[i][k] * t[j];
            }
        }
    }

    vec y(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += A[i][j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    vec z(n);
    for (int i = 0; i < n; ++i) {
        z[i] = y[i] / A[i][i];
    }

    vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[j][i] * x[j];
        }
        x[i] = z[i] - sum;
    }
    
    return x;
}

/**
 * @brief Метод Гаусса без выбора ведущего элемента.
 */
vec solve_gauss_without_chose(matrix A, vec b) {
    int n = A.size();
    
    for (int k = 0; k < n - 1; ++k) {
        if (std::abs(A[k][k]) < 1e-12) { 
            std::cerr << "Ошибка Гаусса (без выбора): Нулевой главный элемент на шаге " << k << ". Метод неприменим." << std::endl;
            return {};
        }
        for (int i = k + 1; i < n; ++i) {
            double lik = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= lik * A[k][j];
            }
            b[i] -= lik * b[k];
        }
    }

    vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}

/**
 * @brief Метод Гаусса с выбором ведущего элемента по столбцу.
 */
vec solve_gauss_with_chose(matrix A, vec b) {
    int n = A.size();

    for (int k = 0; k < n - 1; ++k) {
        int pivot_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > std::abs(A[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            std::swap(A[k], A[pivot_row]);
            std::swap(b[k], b[pivot_row]);
        }

        if (std::abs(A[k][k]) < 1e-12) {
             std::cerr << "Ошибка Гаусса (с выбором): Матрица вырождена." << std::endl;
             return {};
        }

        for (int i = k + 1; i < n; ++i) {
            double lik = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= lik * A[k][j];
            }
            b[i] -= lik * b[k];
        }
    }
    
    vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}


int main() {
    // --- 1. Инициализация входных данных ---
    const int n = 2000;
    const int m = 15; // Пример: номер в списке
    const int k = 4;  // Пример: номер группы

    std::cout << "--- Входные данные ---\n";
    std::cout << "n = " << n << ", m = " << m << ", k = " << k << "\n\n";

    // --- 2. Генерация матрицы A и векторов ---
    matrix A_original(n, vec(n));
    vec x_exact(n);
    vec b_original(n, 0.0);

    std::mt19937 gen(123); // Используем фиксированный seed для воспроизводимости
    std::uniform_real_distribution<> dis(-100.0, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                if (i == 0) A_original[i][j] = std::pow(m + 1, k) + std::pow(m, k);
                else A_original[i][j] = std::pow(m + i + 1, k) + std::pow(m + i, k);
            } else {
                A_original[i][j] = dis(gen);
                A_original[j][i] = A_original[i][j];
            }
        }
    }
    for (int i = 0; i < n; ++i) x_exact[i] = m + i;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_original[i] += A_original[i][j] * x_exact[j];
        }
    }

    // --- 3. Решение и сравнение методов ---
    std::cout << std::fixed << std::setprecision(8);
    
    // Метод 1: LDLT-разложение
    {
        std::cout << "--- 1. Метод LDLT-разложения ---\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        vec x_solution = solve_ldlt(A_original, b_original);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        print_vector(x_solution, "Первые 5 координат:", 5);
        
        // *** ИЗМЕНЕНИЯ ЗДЕСЬ: ВЫВОД НОРМЫ НЕВЯЗКИ И ПОГРЕШНОСТИ ***
        double residual_norm = calculate_residual_norm(A_original, x_solution, b_original);
        std::cout << "Норма вектора невязки: " << residual_norm << std::endl;
        
        vec diff = vector_diff(x_solution, x_exact);
        double relative_error = vector_norm(diff) / vector_norm(x_exact);
        std::cout << "Относительная погрешность: " << relative_error << std::endl;
        std::cout << "Время выполнения: " << elapsed.count() << " секунд\n\n";
    }

    // Метод 2: Гаусс без выбора
    {
        std::cout << "--- 2. Метод Гаусса (без выбора ведущего элемента) ---\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        vec x_solution = solve_gauss_without_chose(A_original, b_original);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        print_vector(x_solution, "Первые 5 координат:", 5);

        // *** ИЗМЕНЕНИЯ ЗДЕСЬ: ВЫВОД НОРМЫ НЕВЯЗКИ И ПОГРЕШНОСТИ ***
        double residual_norm = calculate_residual_norm(A_original, x_solution, b_original);
        std::cout << "Норма вектора невязки: " << residual_norm << std::endl;

        if (!x_solution.empty()) {
            vec diff = vector_diff(x_solution, x_exact);
            double relative_error = vector_norm(diff) / vector_norm(x_exact);
            std::cout << "Относительная погрешность: " << relative_error << std::endl;
        }
        std::cout << "Время выполнения: " << elapsed.count() << " секунд\n\n";
    }

    // Метод 3: Гаусс с выбором
    {
        std::cout << "--- 3. Метод Гаусса (с выбором ведущего элемента) ---\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        vec x_solution = solve_gauss_with_chose(A_original, b_original);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        print_vector(x_solution, "Первые 5 координат:", 5);

        // *** ИЗМЕНЕНИЯ ЗДЕСЬ: ВЫВОД НОРМЫ НЕВЯЗКИ И ПОГРЕШНОСТИ ***
        double residual_norm = calculate_residual_norm(A_original, x_solution, b_original);
        std::cout << "Норма вектора невязки: " << residual_norm << std::endl;

         if (!x_solution.empty()) {
            vec diff = vector_diff(x_solution, x_exact);
            double relative_error = vector_norm(diff) / vector_norm(x_exact);
            std::cout << "Относительная погрешность: " << relative_error << std::endl;
        }
        std::cout << "Время выполнения: " << elapsed.count() << " секунд\n\n";
    }

    return 0;
}