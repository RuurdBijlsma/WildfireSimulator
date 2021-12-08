Bootstrap [boost](https://www.boost.org/users/history/version_1_77_0.html ) depending on compiler version 

```
Visual Studio 2019 -  bootstrap vc142
Visual Studio 2017 -  bootstrap vc141
Visual Studio 2015 -  bootstrap vc140
```

`bootstrap.bat vc141`

configure python location: `[USER]/user-config.jam`:
```
using python 
   : 3.9
   : C:\\Users\\Ruurd\\AppData\\Local\\Programs\\Python\\Python39\\python.exe
   : C:\\Users\\Ruurd\\AppData\\Local\\Programs\\Python\\Python39\\include #directory that contains pyconfig.h
   : C:\\Users\\Ruurd\\AppData\\Local\\Programs\\Python\\Python39\\libs    #directory that contains python39.lib
   ;
```

Build boost python lib

`b2 --with-python --prefix=c:\\boost178 address-model=64 variant=release link=static threading=multi runtime-link=shared install`

copy .pyd to ./cpp folder