#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>

// Псевдонимы типов
using matrix = std::vector<std::vector<double>>;
using vec = std::vector<double>;

// Перечисление типов предобуславливания
enum class PreconditionerType {
    None,       // Обычный CG (M = I)
    Jacobi,     // Якоби (M = diag(A))
    Scaling     // Масштабирование (M = diag(||row_i||))
};

// --- Вспомогательные функции (векторные операции) ---

// Скалярное произведение (sxz)
double dot_product(const vec& a, const vec& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Умножение матрицы на вектор (mvAx)
vec mat_vec_mult(const matrix& A, const vec& x) {
    size_t n = A.size();
    vec result(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Евклидова норма вектора
double vector_norm(const vec& v) {
    return std::sqrt(dot_product(v, v));
}

// Разность векторов: a - b
vec vec_diff(const vec& a, const vec& b) {
    vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
    return res;
}

// Сумма векторов: a + b
vec vec_add(const vec& a, const vec& b) {
    vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}

// Умножение вектора на скаляр: scalar * v
vec vec_scale(double scalar, const vec& v) {
    vec res(v.size());
    for (size_t i = 0; i < v.size(); ++i) res[i] = scalar * v[i];
    return res;
}

// Вывод первых элементов
void print_vector_head(const vec& v, const std::string& title, int count) {
    std::cout << title << std::endl;
    for (int i = 0; i < count && i < (int)v.size(); ++i) {
        std::cout << "  [" << i << "] = " << v[i] << std::endl;
    }
}

// --- Функции для предобуславливания ---

// Создание вектора диагональных элементов матрицы M
vec create_preconditioner(const matrix& A, PreconditionerType type) {
    int n = A.size();
    vec M_diag(n, 1.0); // По умолчанию единицы (для None)

    if (type == PreconditionerType::Jacobi) {
        // M = diag(A) -> берем элементы главной диагонали
        for (int i = 0; i < n; ++i) {
            M_diag[i] = A[i][i]; 
            // Защита от деления на ноль, хотя матрица положительно определенная
            if (std::abs(M_diag[i]) < 1e-15) M_diag[i] = 1.0; 
        }
    } 
    else if (type == PreconditionerType::Scaling) {
        // M = diag(d_ii), где d_ii = евклидова норма i-й строки матрицы A
        for (int i = 0; i < n; ++i) {
            double sum_sq = 0.0;
            for (int j = 0; j < n; ++j) {
                sum_sq += A[i][j] * A[i][j];
            }
            M_diag[i] = std::sqrt(sum_sq);
            if (std::abs(M_diag[i]) < 1e-15) M_diag[i] = 1.0;
        }
    }
    
    return M_diag;
}

// Применение предобуславливателя: решить Mz = r относительно z
// Так как M диагональная, z[i] = r[i] / M_diag[i]
vec apply_preconditioner(const vec& r, const vec& M_diag) {
    int n = r.size();
    vec z(n);
    for (int i = 0; i < n; ++i) {
        z[i] = r[i] / M_diag[i];
    }
    return z;
}

// --- Метод PCG (Preconditioned Conjugate Gradient) ---
vec solve_pcg(const matrix& A, const vec& f, double epsilon, int max_iter, 
              PreconditionerType type, int& iterations_out, 
              double& final_r_norm, double& final_z_norm) {
    int n = f.size();
    vec x(n, 0.0); // x0 = 0
    
    // r0 = f - A*x0 = f
    vec r = f; 
    
    // Подготовка матрицы M (храним только диагональ)
    vec M_diag = create_preconditioner(A, type);

    // z0 = M^(-1) * r0
    vec z = apply_preconditioner(r, M_diag);
    
    // p0 = z0
    vec p = z; 
    
    // (r_0, z_0)
    double rz_curr = dot_product(r, z); 
    
    iterations_out = 0;

    // В методичке сказано проверять норму z (невязка предобусловленной системы)
    // Для обычного CG (None) z совпадает с r.
    final_z_norm = vector_norm(z);
    final_r_norm = vector_norm(r);

    for (int j = 0; j < max_iter; ++j) {
        // Критерий остановки согласно заданию: ||z_{j+1}|| < epsilon
        // Проверяем в начале цикла (для j=0 это ||z0||, далее ||z_{j+1}|| после обновления)
        if (vector_norm(z) < epsilon) {
            break;
        }

        iterations_out++;

        vec Ap = mat_vec_mult(A, p);      // Ap_j
        double Ap_p = dot_product(Ap, p); // (Ap_j, p_j)
        
        double alpha = rz_curr / Ap_p;    // alpha_j
        
        x = vec_add(x, vec_scale(alpha, p));       // x_{j+1}
        r = vec_diff(r, vec_scale(alpha, Ap));     // r_{j+1}
        
        // z_{j+1} = M^(-1) * r_{j+1}
        z = apply_preconditioner(r, M_diag);       // z_{j+1}

        double rz_next = dot_product(r, z);        // (r_{j+1}, z_{j+1})
        
        double beta = rz_next / rz_curr;           // beta_j
        
        p = vec_add(z, vec_scale(beta, p));        // p_{j+1}
        
        rz_curr = rz_next; // Обновляем для следующего шага
    }
    
    final_z_norm = vector_norm(z);
    final_r_norm = vector_norm(r);
    
    return x;
}

int main() {
    // Входные параметры 
    const int n = 2000;
    const int m_student = 15; 
    const int k_group = 4;  
    const int max_iter = 2000; 
    // Эпсилон берем примерно как в Лаб 5 (для double это обычно около 1e-10...1e-13)
    // Зададим жесткий критерий для наглядности разницы в скорости сходимости
    const double epsilon = 1e-11; 

    std::cout << "--- Лабораторная работа №6: Предобусловливание ---\n";
    std::cout << "Параметры: n=" << n << ", m=" << m_student << ", k=" << k_group << "\n";
    std::cout << "Критерий останова (epsilon): " << epsilon << "\n\n";

    // 1. Генерация данных (матрица A, точное решение x_exact, правая часть f)
    matrix A(n, vec(n));
    vec x_exact(n);
    vec f(n);

    std::mt19937 gen(123); 
    std::uniform_real_distribution<> dis(-100.0, 0.0);

    // Заполнение недиагональных элементов
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = dis(gen);
            A[j][i] = A[i][j];
        }
    }
    // Заполнение диагональных (с диагональным преобладанием)
    double sum_first_row = 0.0;
    for (int j = 1; j < n; ++j) sum_first_row += A[0][j];
    A[0][0] = -sum_first_row + 10 * k_group;
    for (int i = 1; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            row_sum += A[i][j];
        }
        A[i][i] = -row_sum + 10 * k_group;
    }

    // Формирование f = A * x_exact
    for (int i = 0; i < n; ++i) x_exact[i] = m_student + i;
    f = mat_vec_mult(A, x_exact);

    std::cout << std::fixed << std::setprecision(8);

    // Список экспериментов
    struct Experiment {
        PreconditionerType type;
        std::string name;
    };
    
    std::vector<Experiment> experiments = {
        {PreconditionerType::None,    "1. Метод CG (без предобуславливания)"},
        {PreconditionerType::Jacobi,  "2. Метод PCG (Якоби)"},
        {PreconditionerType::Scaling, "3. Метод PCG (Масштабирование)"}
    };

    for (const auto& exp : experiments) {
        std::cout << "=== " << exp.name << " ===\n";
        
        int iterations = 0;
        double r_norm = 0.0; // ||r||
        double z_norm = 0.0; // ||z||
        
        auto start = std::chrono::high_resolution_clock::now();
        
        vec x_res = solve_pcg(A, f, epsilon, max_iter, exp.type, 
                              iterations, r_norm, z_norm);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        // Вывод данных
        print_vector_head(x_res, "Первые 5 координат решения x*:", 5);
        std::cout << "Количество итераций: " << iterations << std::endl;
        std::cout << "Норма невязки ||f - Ax*|| (||r||): " << r_norm << std::endl;
        if (exp.type != PreconditionerType::None) {
            std::cout << "Норма предобусл. невязки ||z||:   " << z_norm << std::endl;
        }

        // Относительная погрешность
        double exact_norm = vector_norm(x_exact);
        vec diff_vec = vec_diff(x_res, x_exact);
        double error_norm = vector_norm(diff_vec);
        std::cout << "Относительная погрешность: " << (error_norm / exact_norm) << std::endl;
        
        std::cout << "Время выполнения: " << diff.count() << " с\n\n";
    }

    
    return 0;
}