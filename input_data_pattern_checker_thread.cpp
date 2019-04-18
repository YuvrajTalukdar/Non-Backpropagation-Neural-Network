#include "input_data_pattern_checker_thread.h"
#include<Qtcore>
#include<qdebug.h>

input_data_pattern_checker_thread::input_data_pattern_checker_thread(QObject *parent):QThread(parent)
{}

void input_data_pattern_checker_thread::run()
{
    QMutex mutex;
    mutex.lock();
    while(obj1->core1.body_engine_communication_data_obj.task_complete==false)
    {   if(pause==false)
        {   emit check_pattern();}
        this->msleep(1000);
    }
    mutex.unlock();
}

void input_data_pattern_checker_thread::set_object(data_file_reader_class *obj)
{   obj1=obj;}
