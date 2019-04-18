#ifndef INPUT_DATA_PATTERN_CHECKER_THREAD_H
#define INPUT_DATA_PATTERN_CHECKER_THREAD_H

#include<QThread>
#include<string.h>
#include"data_file_reader_class.h"
#include<iostream>

using namespace std;

class input_data_pattern_checker_thread : public QThread
{
    Q_OBJECT
private:
    data_file_reader_class* obj1;
public:
    explicit input_data_pattern_checker_thread(QObject *parent);
    void run();
    void set_object(data_file_reader_class* obj);
    bool pause=false;
signals:
    void check_pattern();
public slots:

};

#endif // INPUT_DATA_PATTERN_CHECKER_THREAD_H
