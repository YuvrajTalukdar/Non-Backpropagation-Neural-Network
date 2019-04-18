#include "progress_display_thread.h"
#include<Qtcore>
#include<qdebug.h>
progress_display_thread::progress_display_thread(QObject *parent):QThread(parent)
{}

void progress_display_thread::run()
{
    bool sleep_enable=false;
    bool set=false;
    QMutex mutex;
    mutex.lock();
    QString new_str;
    while(obj1->core1.body_engine_communication_data_obj.task_complete==false)
    {
        string str=obj1->core1.body_engine_communication_data_obj.display_message();
        if(obj1->core1.body_engine_communication_data_obj.progress_bar_display==true)
        {   emit update_progress_bar();}
        if(str.compare("")!=0)
        {
            sleep_enable=false;
            QString qstr;//QString::fromUtf8(str.begin(),str.data());
            qstr.append(str.c_str());
            new_str.append(qstr);
            emit progress_display_system_thread(new_str);
            bool display_thread_options_once_emitted=false;
            if(obj1->core1.body_engine_communication_data_obj.thread_selection_point_reached==true && display_thread_options_once_emitted==false)
            {
                emit display_thread_options(true);
                display_thread_options_once_emitted=true;
            }
        }
        else
        {   sleep_enable=true;}
        if(sleep_enable==true)
        {   this->msleep(500);}

        if(set==false && obj1->core1.body_engine_communication_data_obj.prediction_on_individual_data_function_reached==true)
        {
            set=true;
            emit set_label();
        }
    }
    if(obj1->core1.body_engine_communication_data_obj.task_complete==true)
    {
        emit task_complete_signal();
    }    

    mutex.unlock();
}

void progress_display_thread::set_object(data_file_reader_class *obj)
{   obj1=obj;}
