#include "core_thread.h"
#include<Qtcore>
#include<qdebug.h>

core_thread::core_thread(QObject *parent):QThread(parent)
{}

void core_thread::run()
{
    QMutex mutex;
    mutex.lock();
    obj1->core_starter(data_set_file_name,train_test_predict,data_division,network_save_file_name);
    mutex.unlock();
}

void core_thread::set_object_and_data(data_file_reader_class *obj,string dataset,string network_file,int radio_button_option,float data_div)
{
    obj1=obj;
    data_set_file_name=dataset;
    network_save_file_name=network_file;
    train_test_predict=radio_button_option;
    data_division=data_div;
}
