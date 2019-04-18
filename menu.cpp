#include<iostream>
#include<string.h>
#include<dirent.h>
#include<vector>
#include<iomanip>//for erasing the above line
#include<sys/ioctl.h>//for the terminal size

//#include"core_start.h"

using namespace std;

bool menu(string &file_name,int &test_train_predict,float &data_division,string &network_save_file_name)
{
    //.csv file finder
    struct dirent *de;  // Pointer for directory entry
    // opendir() returns a pointer of DIR type. 
    DIR *dr = opendir(".");
    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        cout<<"Could not open current directory";
        return false;
    }
    char filename_sub_str[]=".csv";
    vector<string> csv_save_file;
    csv_save_file.clear();
    while ((de = readdir(dr)) != NULL)
    {
        if(strcasestr(de->d_name,filename_sub_str) && strcasestr(de->d_name,"network-")==NULL)//strcasestr returns only char*
        {   csv_save_file.push_back(de->d_name);}
    }
    closedir(dr);
    //home page
    struct winsize w;
    ioctl(0,TIOCGWINSZ,&w);
    point0:
    system("clear");
    string heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
    cout<<setw(w.ws_col/2+heading.length()/2)<<heading;
    cout<<"\n\n1. Train network on the entire dataset\n"<<"2. Train and test the network\n"<<"3. Use a trained network to make predictions\n";
    cout<<"4. Enter individual data and predict the results using a trained network\n";
    cout<<"\nPress 0 to close\n\n";
    point1:
    cout<<"Select your option: ";
    int option1;
    cin>>option1;
    if(option1>4 || option1<0)
    {
        cout<<"Wrong option!!!\n";
        goto point1;
    }
    else if(option1==1)
    {
        //file selector for complete training
        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;      
        string note="\n\nHere a new network will be created if no network file exist and trained on the entire dataset, \nelse if a network file is available than a new network file will be creeated which will have \nall the connections from the old network and also the new connections will be created this time, so \nif you want to test the accuracy of the network generated, it is scientifically not accurate to \ntest the network trained on the same dataset so better do the testing on another new dataset of \nsame nature. The accuracy may be increased in the process of multiple training.\nIF YOU WANT A BRAND NEW NETWORK THAN MAKE SURE THE CURRENT DIRECTORY DONT WAVE ANY NETWORK FILE.";
        cout<<"\n\nNOTE:-"<<note;  
        cout<<"\n\nList of available csv files for input data in the current directory:-\n";
        for(int a=0;a<csv_save_file.size();a++)
        {   cout<<a+1<<". "<<csv_save_file[a]<<endl;}   
        if(csv_save_file.size()==0)
        {   cout<<"\nERROR No .csv file found in the current directory!!!\n";}
        cout<<"\nPress 0 to go back"; 
        int file_choice;
        point2:
        cout<<"\n\nSelect your option: ";
        cin>>file_choice;
        if(file_choice>csv_save_file.size() || file_choice<0)
        {   
            cout<<"wrong option!!!";
            goto point2;
        }
        else if(file_choice==0)
        {   
            system("clear");
            goto point0;
        }
        else
        {   file_name=csv_save_file[file_choice-1];}
        //setting test_train_predict
        test_train_predict=0;//creates a brand new trained network
    }
    else if(option1==2)
    {
        //file selector for part training and part testing
        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;      
        string note1="\n\nThis will create a new network trained with specific test : train ratio dataset.\nThe dataset will be divided into testing and training sets, and after training \nthe network on the training set its acuracy will be automatically tested \non the testing set.";
        cout<<"\n\nNOTE:-"<<note1;  
        cout<<"\n\nList of available csv files for input data in the current directory:-\n";
        for(int a=0;a<csv_save_file.size();a++)
        {   cout<<a+1<<". "<<csv_save_file[a]<<endl;}   
        if(csv_save_file.size()==0)
        {   cout<<"\nERROR No .csv file found in the current directory!!!\n";}
        cout<<"\nPress 0 to go back"; 
        int file_choice;
        point3:
        cout<<"\n\nSelect your option: ";
        cin>>file_choice;
        if(file_choice>csv_save_file.size() || file_choice<0)
        {   
            cout<<"wrong option!!!";
            goto point3;
        }
        else if(file_choice==0)
        {   
            system("clear");
            goto point0;
        }
        else
        {   file_name=csv_save_file[file_choice-1];}
        //data division selector
        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;        
        cout<<"\n\nEnter the test : train ratio :-\n\n";
        cout<<"(test data size=total data / divisor,train data = total data - test data, \nin case of cancer dataset dont go beyond 21 and in case of space suttle dataset dont go beyond 1.5)\n\n";
        cout<<"Enter data_division value: ";
        cin>>data_division;
        test_train_predict=1;
    }
    else if(option1==3)
    {
        point7:
        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        string note2="\n\nThe prediction results are saved on a file named prediction_results.csv. ";
        cout<<"\n\nNOTE:-"<<note2; 
        cout<<"\n\nChoose the network save file:-\n";
        //.network file finder
        struct dirent *de2;  // Pointer for directory entry
        // opendir() returns a pointer of DIR type. 
        DIR *dr2 = opendir(".");
        if (dr2 == NULL)  // opendir returns NULL if couldn't open directory
        {
            cout<<"Could not open current directory";
            return false;
        }
        char filename_sub_str2[]="network-";
        vector<string> network_save_file;
        network_save_file.clear();
        while ((de2 = readdir(dr2)) != NULL)
        {
            if(strcasestr(de2->d_name,filename_sub_str2))
            {   network_save_file.push_back(de2->d_name);}
        }
        closedir(dr2);
        for(int a=0;a<network_save_file.size();a++)
        {   cout<<a+1<<". "<<network_save_file[a]<<endl;}
        if(network_save_file.size()==0)
        {   cout<<"\nERROR No network file found in the current directory!!!\n";}
        cout<<"\npress 0 to go back\n";
        point4:
        cout<<"\nEnter your option: ";
        int option2;
        cin>>option2;//fddfsffsfsdfdsf
        if(option2>network_save_file.size() || option2<0)
        {
            cout<<"Wrong option!!!";
            goto point4;
        }
        else if(option2==0)
        {   goto point0;}
        else
        {   network_save_file_name=network_save_file[option2-1];}
        //data_file_selector

        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;     
        string note1="\n\nSelect the data file for the network you have selected before. If the selected datafile's data is \nnot of the type which was used to train the network in the network file the software may crash or \nstop. ";
        cout<<"\n\nNOTE:-"<<note1;  
        cout<<"\n\nList of available csv files for input data in the current directory:-\n";
        for(int a=0;a<csv_save_file.size();a++)
        {   cout<<a+1<<". "<<csv_save_file[a]<<endl;}   
        if(csv_save_file.size()==0)
        {   cout<<"\nERROR No .csv file found in the current directory!!!\n";}
        cout<<"\nPress 0 to go back"; 
        int file_choice;
        point6:
        cout<<"\n\nSelect your option: ";
        cin>>file_choice;
        if(file_choice>csv_save_file.size() || file_choice<0)
        {   
            cout<<"wrong option!!!";
            goto point6;
        }
        else if(file_choice==0)
        {   
            system("clear");
            goto point7;
        }
        else
        {   file_name=csv_save_file[file_choice-1];}

        test_train_predict=2;
    }
    else if(option1==4)
    {
        system("clear");
        heading="NON BACKPROPAGATION ALGORITHM TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        string note2="\n\nHere you can enter the data yourself and let the algorithm predict the appropriate result, \nthe result will appear instantaneously in this interface itself.";
        cout<<"\n\nNOTE:-"<<note2; 
        cout<<"\n\nChoose the network save file:-\n";
        //.network file finder
        struct dirent *de2;  // Pointer for directory entry
        // opendir() returns a pointer of DIR type. 
        DIR *dr2 = opendir(".");
        if (dr2 == NULL)  // opendir returns NULL if couldn't open directory
        {
            cout<<"Could not open current directory";
            return false;
        }
        char filename_sub_str2[]="network-";
        vector<string> network_save_file;
        network_save_file.clear();
        while ((de2 = readdir(dr2)) != NULL)
        {
            if(strcasestr(de2->d_name,filename_sub_str2))
            {   network_save_file.push_back(de2->d_name);}
        }
        closedir(dr2);
        for(int a=0;a<network_save_file.size();a++)
        {   cout<<a+1<<". "<<network_save_file[a]<<endl;}
        cout<<"\npress 0 to go back\n";
        point5:
        cout<<"\nEnter your option: ";
        int option2;
        cin>>option2;
        if(option2>network_save_file.size() || option2<0)
        {
            cout<<"Wrong option!!!";
            goto point5;
        }
        else if(option2==0)
        {   goto point0;}
        else
        {   network_save_file_name=network_save_file[option2-1];}
        test_train_predict=3;//a new option 
    }
    else if(option1==0)
    {   test_train_predict=-1;}
}