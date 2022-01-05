#include "application.h"


int main()
{
    Application app;
    app.Init(true);
    app.Run();
    app.Cleanup();
    return 0;
}