#include "utils/string_utils.h"
#include <sstream>
#include <algorithm>
#include <string.h>


std::vector<std::string> StringUtils::Split(
    const std::string& str, const std::vector<char>& delim) 
{
    std::vector<std::string> words;
    std::stringstream ss;
    for (char c : str) {
        if (std::find(delim.begin(), delim.end(), c) != delim.end()) {
            std::string word = ss.str();
            if (word != "") {
                words.push_back(word);
            }
            ss.str("");
        }
        else {
            ss << c;
        }
    }
    std::string word = ss.str();
    if (word != "") {
        words.push_back(word);
    }
    return words;
}

std::string StringUtils::Replace(
    const std::string& str, const std::string& a, const std::string& b) 
{
    std::stringstream ss;
    for (size_t i = 0; i < str.size();) {
        if (i + a.size() <= str.size() && memcmp(str.c_str() + i, a.c_str(), a.size()) == 0) {
            ss << b;
            i += a.size();
        }
        else {
            ss << str[i];
            i++;
        }
    }
    return ss.str();
}