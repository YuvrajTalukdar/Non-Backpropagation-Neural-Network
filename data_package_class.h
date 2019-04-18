/*
Data package class
*/

#include<vector>
#include<iostream>

using namespace std;

class data_package_class{
    public:
    
    vector<vector<float>> data;  
    vector<int> labels; 
    bool analyze_status=false;
    
    void set_no_of_record(int n)
    {
        data.resize(n);
        labels.resize(n);
    }

    void set_no_of_elements_in_each_record(int n)
    {
        for(int a=0;a<data.size();a++)
        {
            data[a].resize(n);
        }
    }
    
    int no_fo_records()
    {
        return data.size();
    }

    int no_of_elements_in_each_record()
    {
        return data[0].size();
    }
};

struct filtered_data
{
    vector<vector<float>> data;
    int label;
};