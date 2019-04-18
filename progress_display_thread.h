#ifndef PROGRESS_DISPLAY_THREAD_H
#define PROGRESS_DISPLAY_THREAD_H

#include<QThread>
#include<string.h>
#include"data_file_reader_class.h"
#include<iostream>
using namespace std;

class progress_display_thread : public QThread
{
    Q_OBJECT
private:
    data_file_reader_class* obj1;
public:
    explicit progress_display_thread(QObject *parent);
    void run();
    bool Stop;
    void set_object(data_file_reader_class* obj);
signals:
    void progress_display_system_thread(QString);
    void display_thread_options(bool);
    void task_complete_signal();
    void update_progress_bar();
    void set_label();
public slots:

};

#endif // PROGRESS_DISPLAY_THREAD_H
