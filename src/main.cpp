#include "application.h"


int main()
{
    Application app;
    app.Init(false);
    app.Run();
    app.Cleanup();
    return 0;
}