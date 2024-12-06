//Writting header fences like some kind of barbarian ~MK
#pragma once
#ifndef BNSHACKS_H
#define BNSHACKS_H
#include <iostream>
//Mechanism from cudaSafeCall

#define bnsSafeCall(err) __bnsSafeCall(err,__FILE__,__LINE__)
inline void __bnsSafeCall(int err,const char *file, const int line)
{
	if (err == false) // if false there was an error
	{
		std::cout << "bnsSafeCall() failed with code: "<<err <<" at line "<<line<<": "<<file<<std::endl;
		exit(-1);
	}
	return;
}
//TODO: write a check and printf function
#endif