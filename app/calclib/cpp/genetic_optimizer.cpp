// genetic_optimizer.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

//#include <iostream>
#include "main.h"
#include <nlohmann/json.hpp>
#include <fstream>

/*
void local_func(std::unordered_map<size_t,size_t>& dict)
{
    size_t i = 0;
    while(i<10)
    {
        dict[i] = i * i;
        ++i;
    }
}bool compare_(std::pair<size_t, float > A, std::pair<size_t, float > B )
{
    if (A.second < B.second) { return true; }
    else return false;
}

static void print_bits(const char* little_endian_memory, size_t size)
{
    for (size_t i = size; i-- > 0; ) {
        const unsigned char byte = little_endian_memory[i];
        for (size_t n = CHAR_BIT; n-- > 0; )
            putchar(byte & (1u << n) ? '1' : '0');
    }
    std::cout << "\n";
}
*/
int main()
{
    /*int temp;
    temp = _CRTDBG_ALLOC_MEM_DF;
    std::cout << "CRTDBG_ALLOC_MEM_DF " << temp << "\n";    
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_DELAY_FREE_MEM_DF;
    std::cout << "_CRTDBG_DELAY_FREE_MEM_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_CHECK_ALWAYS_DF;
    std::cout << "_CRTDBG_CHECK_ALWAYS_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_CHECK_CRT_DF;
    std::cout << "_CRTDBG_CHECK_CRT_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_LEAK_CHECK_DF;
    std::cout << "_CRTDBG_LEAK_CHECK_DF" << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);

    int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    std::cout << tmpFlag << "\n";
    print_bits((const char*)&tmpFlag, sizeof tmpFlag);
    // Turn On (OR) - Keep freed memory blocks in the
    // heap's linked list and mark them as freed
    tmpFlag |= _CRTDBG_DELAY_FREE_MEM_DF;
    std::cout << tmpFlag << "\n";
    print_bits((const char*)&tmpFlag, sizeof tmpFlag);
    // Turn Off (AND) - prevent _CrtCheckMemory from
    // being called at every allocation request
    tmpFlag &= ~_CRTDBG_CHECK_ALWAYS_DF;
    std::cout << tmpFlag << "\n";
    print_bits((const char*)&tmpFlag, sizeof tmpFlag);
    // Set the new state for the flag
    _CrtSetDbgFlag(tmpFlag);
    std::cout << _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) << "\n";

    //int temp;
    temp = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    print_bits((const char*)&temp, sizeof temp);
    std::cout << temp <<" " << _CrtSetDbgFlag(temp) << "\n";
    temp = temp | _CRTDBG_CHECK_ALWAYS_DF;    
    std::cout << temp << " " << _CrtSetDbgFlag(temp) << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = temp | _CRTDBG_LEAK_CHECK_DF;
    std::cout << temp << " " << _CrtSetDbgFlag(temp) << "\n";
    print_bits((const char*)&temp, sizeof temp);

    //temp = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    std::cout << temp << " " << _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) << "\n";

    std::cout << _CRTDBG_CHECK_ALWAYS_DF<<"\n";
    std::cout << _CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF) << "\n";
    std::cout << _CRTDBG_CHECK_ALWAYS_DF << "\n";
    std::cout << _CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF) << "\n";
    std::cout << _CRTDBG_LEAK_CHECK_DF << "\n";
    std::cout << _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF) << "\n";
    std::cout << _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG)<< "\n";
    std::cout << (_CRTDBG_CHECK_ALWAYS_DF| _CRTDBG_LEAK_CHECK_DF) << "\n";
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF);
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF );*/

    /*
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    //_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    //_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
    int temp;
    temp = _CRTDBG_ALLOC_MEM_DF;
    std::cout << "CRTDBG_ALLOC_MEM_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_DELAY_FREE_MEM_DF;
    std::cout << "_CRTDBG_DELAY_FREE_MEM_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_CHECK_ALWAYS_DF;
    std::cout << "_CRTDBG_CHECK_ALWAYS_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_CHECK_CRT_DF;
    std::cout << "_CRTDBG_CHECK_CRT_DF " << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);
    temp = _CRTDBG_LEAK_CHECK_DF;
    std::cout << "_CRTDBG_LEAK_CHECK_DF" << temp << "\n";
    print_bits((const char*)&temp, sizeof temp);

   
    temp = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    std::cout << "set flag"<<temp  << "\n";
    print_bits((const char*)&temp, sizeof temp);
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF);
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    std::cout << "current" << _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) << "\n";
    //std::random_device rd;
    //Individ* array = new Individ(10, 15, rd);
    //Individ* array_ = new Individ(10, 15, rd);
    // Выделяем память для массива из 5 целых чисел
    int* array = new int[5];
    
    // Заполняем массив значениями
    for (int i = 0; i < 5; ++i) {
        array[i] = i;
    }



    //array[0] = 42;
    // Попытка записи за пределы выделенного блока (ошибка!)
    array[5] = 42;  // Ошибка: выход за пределы массива
    
    
    // Проверка состояния памяти
    if (!_CrtCheckMemory()) {
        std::cerr << "Error.\n";
    }
    else {
        std::cout << "fine.\n";
    }
    
    delete[] array;
    */

    std::string jpath = "//home//duryagin//examp-cpp//data.json";
    using json = nlohmann::json;
    std::ifstream f(jpath);
    json data = json::parse(f);

    using uniform_real = std::uniform_real_distribution<float>;
    std::mt19937 random_generator_;
    uniform_real random_;
    float a = 1.96e5,b=5.9e6;
    float a_ = 1.8e6,b_=3.6e6;
    size_t size = 0;
    json::iterator iter = data["lost"].begin();
    while (iter != data["lost"].end()) {
        
        ++size;
        ++iter;
    }
    size_t npopul_ = 1000, length_ = size, ncode_ = size, mutate_cell_ = 5, cast_number_ = 3;
    float threshold_ = 0.7f,inh_threshold_=0.1f, epsilon_ = 1e-3f, tolerance_ = 0.7f;
    float* x = new float[length_];
    float* y = new float[length_];
    float bound_ = 0.f;
    size_t i = 0;
    iter = data["lost"].begin();
    while (iter != data["lost"].end()) {
        int v = std::stoi(iter.key());
        x[v] = iter.value();
        bound_ += x[v];
        ++i;
        ++iter;
    }
    i = 0;
    iter = data["price"].begin();
    while (iter != data["price"].end()) {
        int v = std::stoi(iter.key());
        y[v] = iter.value();
        ++i;
        ++iter;
    }

    bound_ = bound_ * 0.1;
    //bound_ = 15.f;
    //_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    //_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    //_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
    //_CrtSetDbgFlag((_CRTDBG_ALLOC_MEM_DF | _CRTDBG_DELAY_FREE_MEM_DF) & ~_CRTDBG_CHECK_ALWAYS_DF);
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF| _CRTDBG_DELAY_FREE_MEM_DF| _CRTDBG_LEAK_CHECK_DF);
    //Individ individ = Individ(10, 15., rd);
    Population population = Population(npopul_, bound_, x, y, length_, threshold_, inh_threshold_,epsilon_, tolerance_, mutate_cell_, cast_number_,true); 
    //PopulationParallel population = PopulationParallel(npopul_, bound_, x, y, length_, threshold_, inh_threshold_, epsilon_, tolerance_, mutate_cell_, cast_number_, false,9);
    population.optimize(3);
    individ_ptr pointer = population.optimal_individ;
    Individ* individ = pointer.get();

    std::cout << "val " << individ->get_val() << "  rest " << individ->get_rest() << "\n";
    //std::cout << "\n";
    i = 0;
    while (i < length_)
    {
        std::cout <<i<<" "<< individ->at(i) << " " << x[i] << " " << y[i] << "\n";
        ++i;
    }
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
