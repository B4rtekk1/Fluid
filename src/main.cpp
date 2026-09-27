#include "App.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv)
{
    try {
        if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--smoke")) {
            std::cerr << "Usage: Fluid.exe [--smoke]\n";
            return 2;
        }
        App app;
        app.run(argc == 2 ? 120 : 0);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fluid: " << error.what() << '\n';
        return 1;
    }
}
