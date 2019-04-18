#ifndef DATA_FILE_READER_CLASS_H
#define DATA_FILE_READER_CLASS_H
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<string.h>
#include<algorithm>

//#include"data_package_class.h"
#include"core_class.h"

using namespace std;

class data_file_reader_class
{
    private:
    string file_name;
    struct raw_data{
        vector<vector<float>> rawData;
    };
    bool check_for_missing_data(string line) //to remove the missing data.
    {
        bool status=false;
        for(int a=0;a<line.size();a++)
        {
            if(line.at(a)=='?')
            {
                status=true;
            }
        }
        return status;
    }

    void breaker(raw_data* rw_data,string line)
    {
        char num_char[20]={'\0'},ch[2];
        vector<float> one_row_of_data;
        if(check_for_missing_data(line)==false)
        {
            for(int a=0;a<line.size();a++)
            {
                if(line.at(a)==',')
                {
                    float val = atof(num_char);
                    //cout<<val<<endl;
                    one_row_of_data.push_back(val);
                    for(int b=0;b<20;b++){
                        num_char[b]='\0';
                    }
                    continue;
                }
                ch[0]=line.at(a);
                ch[1]='\0';
                strcat(num_char,ch);
            }
            one_row_of_data.push_back(atof(num_char));
            rw_data->rawData.push_back(one_row_of_data);
        }
    }

    void read_data_from_file(raw_data* rw_data)
    {
        ifstream  raw_data_stream(file_name,ios::in);
        string line;
        //int count=0;
        while(raw_data_stream)
        {
            raw_data_stream>>line;
            if(raw_data_stream.eof()){
                break;
            }
            //to remove the id line. //deleted if(count>0)
            breaker(rw_data,line);
            //count++; //for just counting the no of lines in the file //required if if(count>0) line to remove the id line
            //cout<<line<<endl;
        }
    }

    void display_rw_data(raw_data* rw_data)
    {
        for(int a=0;a<rw_data->rawData.size();a++)
        {
            for(int b=0;b<rw_data->rawData[a].size();b++)
            {
                cout<<rw_data->rawData[a][b]<<",";
            }
            cout<<endl;
        }
    }

    void data_filter(raw_data* filtered_data)
    {
        raw_data rw_data;
        read_data_from_file(&rw_data);
        //display_rw_data(&rw_data);
        vector<float> row;
        for(int a=0;a<rw_data.rawData.size();a++)
        {
            row.clear();
            for(int b=0;b<rw_data.rawData[a].size();b++)
            {
                //b>0 option removes the id column //deleted if(b>0)
                row.push_back(rw_data.rawData[a][b]);
            }
            filtered_data->rawData.push_back(row);
        }
    }

    void prepare_data(data_package_class* data_pack)
    {
        raw_data filtered_data;
        data_filter(&filtered_data);
        //display_rw_data(&filtered_data);

        vector<float> row;
        //this part pushes the data in the data vector(vector<vector<float>>) and lebel(vector<float>) in the label vector
        for(int a=0;a<filtered_data.rawData.size();a++)
        {
            row.clear();
            for(int b=0;b<filtered_data.rawData[a].size();b++)
            {
                if(b==filtered_data.rawData[a].size()-1)
                {
                    data_pack->labels.push_back(filtered_data.rawData[a][b]);
                }
                else
                {
                    row.push_back(filtered_data.rawData[a][b]);
                }
            }
            data_pack->data.push_back(row);
        }
    }

    void display_prepared_data(data_package_class* data_pack)
    {cout<<"\n\nhello= "<<file_name<<endl;;
        for(int a=0;a<data_pack->data.size();a++)
        {
            //if(data_pack->labels[a]==8){ //for checking each of the labels
            for(int b=0;b<data_pack->data[a].size();b++)
            {
                cout<<data_pack->data[a][b]<<",";
            }
            cout<<" label= "<<data_pack->labels[a]<<endl;
            //}
        }
    }

    core_class* core_address;
    public:
    core_class core1;

    void reinitialize_the_core()
    {
        //delete &core1;
        core_address=new core_class();
        //free(core_address);
        core1=*core_address;
    }

    void core_starter(string &file_name_local,int &test_train_predict,float &data_division,string &network_save_file_name)
    {
        data_package_class data_pack;
        file_name=file_name_local;
        if(test_train_predict==0 || test_train_predict==1 || test_train_predict==2)
        {
            prepare_data(&data_pack);
            cout<<"data file reading success!!!\n";
        }
        //display_prepared_data(&data_pack);

        core1.core_data_entry_point(&data_pack,test_train_predict,data_division,network_save_file_name);

        //reinitialize_the_core();
    }
};

#endif // DATA_FILE_READER_CLASS_H

