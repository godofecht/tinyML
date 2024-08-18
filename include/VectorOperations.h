//Maybe move all of this to a vectorOperations.h file somewhere.
//Vector and float operations
inline std::vector<float> operator- (std::vector<float> a, float b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back  (a[i] - b);
    }
    return retvect;
}

inline std::vector<float> operator+ (std::vector<float> a, float b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back  (a[i] + b);
    }
    return retvect;
}

inline std::vector<float> operator* (std::vector<float> a, float b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back  (a[i] * b);
    }
    return retvect;
}

inline std::vector<float> operator/ (std::vector<float> a, float b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back  (a[i] / b);
    }
    return retvect;
}


//Vector and Vector Operations
inline std::vector<float> operator- (std::vector<float> a, std::vector<float> b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back  (a[i] - b[i]);
    }
    return retvect;
}

inline std::vector<float> operator* (std::vector<float> a, std::vector<float> b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size() ; i++)
    {
        retvect.push_back (a[i] * b[i]);
    }
    return retvect;
}

inline std::vector<float> operator/ (std::vector<float> a, std::vector<float> b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size() ; i++)
    {
        retvect.push_back (a[i] / b[i]);
    }
    return retvect;
}

inline std::vector<float> operator+ (std::vector<float> a, std::vector<float> b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size() ; i++)
    {
        retvect.push_back (a[i] + b[i]);
    }
    return retvect;
}