#ifndef CORE_THREAD_H
#define CORE_THREAD_H

#include<QThread>
#include<string.h>
#include"data_file_reader_class.h"
#include<iostream>

using namespace  std;

class core_thread : public QThread
{   
    Q_OBJECT
private:
    data_file_reader_class* obj1;
    string data_set_file_name,network_save_file_name;
    int train_test_predict;
    float data_division;

public:
    explicit core_thread(QObject *parent);
    void run();
    bool Stop;
    void set_object_and_data(data_file_reader_class*,string,string,int,float);
signals:
    //void progress_display_system_thread(QString);
public slots:
};

#endif // CORE_THREAD_H
