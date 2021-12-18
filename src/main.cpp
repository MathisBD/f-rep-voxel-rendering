#include "application.h"


int main()
{
    Application app;
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}