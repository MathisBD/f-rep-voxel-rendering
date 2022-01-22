#pragma once
#include <string>
#include <vector>


class StringUtils 
{
public:
    // Returns a copy of str in which every occurence of a was replaced by b.
    static std::string Replace(
        const std::string& str, const std::string& a, const std::string& b);

    static std::vector<std::string> Split(
        const std::string& str, const std::vector<char>& delim);
};