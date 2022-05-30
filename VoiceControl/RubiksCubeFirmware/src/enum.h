#ifndef ENUM_H
#define ENUM_H

enum class dir : bool
{
    cw = false,
    ccw = true,
    open = false,
    close = true
};
enum class state
{
    unlock1,
    flip1,
    lock1,
    unlock2,
    flip2,
    lock2,
    turn,
    unlock3,
    flip3,
    turn_back,
    lock3,
    unlock4,
    flip4,
    lock4
};
enum class cube_sides
{
    U,
    D
};

#endif