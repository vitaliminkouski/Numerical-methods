#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Константы
const int N = 4;            // Порядок матрицы
const float EPS = 1e-8;     // Точность для сравнения с нулем

// Тип данных для матрицы
typedef vector<vector<float>> Matrix;

// Функция для создания единичной матрицы
Matrix createIdentity(int n) {
    Matrix res(n, vector<float>(n, 0.0));
    for (int i = 0; i < n; i++)
        res[i][i] = 1.0;
    return res;
}

// Функция для вывода матрицы на экран
void printMatrix(const string& title, const Matrix& m) {
    cout << title << ":" << endl;
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            // Форматированный вывод (ширина 10, 4 знака после запятой)
            cout << setw(10) << fixed << setprecision(4) << m[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Функция перемножения двух матриц: C = A * B
Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<float>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Функция вычисления следа матрицы (сумма диагональных элементов)
float getTrace(const Matrix& A) {
    float sum = 0;
    for (int i = 0; i < A.size(); i++) {
        sum += A[i][i];
    }
    return sum;
}

// Основная логика метода Данилевского
// Возвращает false, если попался нерегулярный случай (деление на 0), иначе true
bool danilevskyMethod(Matrix A, Matrix& Phi, vector<Matrix>& M_list, vector<float>& p) {
    int n = A.size();
    Phi = A; // Начинаем с исходной матрицы, будем ее преобразовывать
    M_list.clear();

    // Алгоритм идет от k = n-1 до 1 (в индексации 0..n-1)
    // Это соответствует описанию PDF: "на (n-k+1)-м шаге, k=n, n-1...2"
    // Мы хотим привести строку i (где i меняется от n-1 до 1) к каноническому виду.
    // Ведущий элемент для строки i находится в столбце i-1.
    
    for (int i = n - 1; i > 0; i--) {
        // Ведущий элемент: Phi[i][i-1]
        float pivot = Phi[i][i - 1];

        // Проверка на регулярность случая
        if (fabs(pivot) < EPS) {
            cout << "Нерегулярный случай: элемент a[" << i << "][" << i - 1 << "] слишком мал (" << pivot << ")." << endl;
            return false;
        }

        // 1. Создаем матрицу M_{i-1} (в коде обозначим как M)
        // Она единичная, кроме строки с индексом (i-1)
        Matrix M = createIdentity(n);
        
        // Заполняем особенную строку (i-1) в матрице M
        // Формулы из PDF (стр. 3): 
        // На месте (i-1, i-1) ставится 1 / a_{i, i-1}
        // На остальных местах (i-1, j) ставится -a_{i, j} / a_{i, i-1}
        for (int j = 0; j < n; j++) {
            if (j == i - 1) {
                M[i - 1][j] = 1.0 / pivot;
            } else {
                M[i - 1][j] = -Phi[i][j] / pivot;
            }
        }

        // 2. Создаем обратную матрицу M^{-1} (обозначим M_inv)
        // Она единичная, кроме строки (i-1), которая совпадает с i-й строкой текущей матрицы A (Phi)
        Matrix M_inv = createIdentity(n);
        for (int j = 0; j < n; j++) {
            M_inv[i - 1][j] = Phi[i][j];
        }

        // 3. Выполняем преобразование подобия: A_{new} = M^{-1} * A_{old} * M
        // Сначала умножаем слева: Temp = M_inv * Phi
        Matrix Temp = multiply(M_inv, Phi);
        // Потом умножаем справа: Phi = Temp * M
        Phi = multiply(Temp, M);

        // Сохраняем матрицу M (понадобится для собственных векторов по заданию)
        // В задании просят сохранить M_{n-1}, M_{n-2}...
        M_list.push_back(M);
    }

    // Извлекаем коэффициенты p1, p2... pn из первой строки полученной матрицы Фробениуса
    // В форме Фробениуса (как в PDF) первая строка содержит p1, p2, ..., pn
    p.clear();
    for (int j = 0; j < n; j++) {
        p.push_back(Phi[0][j]);
    }

    return true;
}

int main() {
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0))); // Инициализация генератора случайных чисел

    Matrix A(N, vector<float>(N));
    Matrix Phi;
    vector<Matrix> M_list;
    vector<float> p_coeffs;
    bool success = false;

    cout << "=== Лабораторная работа №7: Метод Данилевского ===" << endl;
    cout << "Порядок матрицы n = " << N << endl;
    cout << "Диапазон чисел: от -50 до 50" << endl;
    cout << "Тип данных: float" << endl << endl;

    // Цикл генерации матрицы, пока не попадется регулярный случай
    do {
        // Заполнение матрицы случайными числами
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // rand() % 101 дает 0..100, вычитаем 50 -> -50..50
                // добавляем дробную часть для float
                float randomVal = (rand() % 101) - 50.0f; 
                // Можно добавить небольшую дробную часть, чтобы числа были не только целыми
                randomVal += static_cast<float>(rand()) / RAND_MAX;
                A[i][j] = randomVal;
            }
        }

        cout << "Сгенерирована новая матрица A." << endl;
        
        // Попытка приведения к форме Фробениуса
        success = danilevskyMethod(A, Phi, M_list, p_coeffs);
        
        if (!success) {
            cout << "Генерация новой матрицы...\n" << endl;
        }

    } while (!success);

    // === ВЫВОД РЕЗУЛЬТАТОВ ===

    // 1. Исходная матрица
    printMatrix("Исходная матрица A", A);

    // 2. Матрицы M (выводим в порядке их получения: M_{n-1}, M_{n-2}, M_1)
    // В нашем списке они лежат в порядке добавления. Первая добавленная соответствовала i=n-1.
    cout << "Матрицы преобразования (M_{n-1}, ..., M_1):" << endl;
    for (size_t i = 0; i < M_list.size(); i++) {
        string name = "M_" + to_string(N - 1 - i); // Формируем имя M_3, M_2...
        printMatrix(name, M_list[i]);
    }

    // 3. Каноническая форма Фробениуса
    printMatrix("Каноническая форма Фробениуса (Phi)", Phi);

    // 4. Коэффициенты p
    cout << "Коэффициенты характеристического многочлена (p):" << endl;
    for (size_t i = 0; i < p_coeffs.size(); i++) {
        cout << "p" << (i + 1) << " = " << p_coeffs[i] << endl;
    }
    cout << endl;

    // 5. Проверка (След матрицы)
    float traceA = getTrace(A);
    float p1 = p_coeffs[0]; // p1 - это первый элемент первой строки формы Фробениуса

    cout << "=== Контроль вычислений ===" << endl;
    cout << "След исходной матрицы Sp(A) = " << traceA << endl;
    cout << "Коэффициент p1              = " << p1 << endl;
    cout << "Разность |Sp(A) - p1|       = " << fabs(traceA - p1) << endl;

    if (fabs(traceA - p1) < 1e-3) { // Допуск чуть больше EPS из-за накопления ошибок float
        cout << "Проверка пройдена: Sp(A) приблизительно равен p1." << endl;
    } else {
        cout << "Внимание: значительная погрешность вычислений." << endl;
    }

    return 0;
}