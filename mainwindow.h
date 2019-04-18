#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<iostream>
#include<thread>
#include<QString>
#include<QThread>
#include<string>
#include<qdebug.h>
#include<QtCore>
#include<QCloseEvent>

#include"data_file_reader_class.h"
#include"progress_display_thread.h"
#include"core_thread.h"
#include<input_data_pattern_checker_thread.h>
using namespace std;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    progress_display_thread *pds_thread;
    core_thread *c_thread;
    input_data_pattern_checker_thread *idpc_thread;
    void closeEvent(QCloseEvent *event);
public slots:
    void progress_display_system(QString);
    void on_display_thread_options(bool);
    void on_task_complete_signal();
    void on_update_progress_bar();
    void on_set_label();
private slots:
    void on_radioButton_1_clicked();

    void on_radioButton_2_clicked();

    void on_radioButton_3_clicked();

    void on_radioButton_4_clicked();

    void on_radioButton_5_clicked();

    void on_radioButton_6_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_1_clicked();

    void on_pushButton_2_clicked();

    void on_lineEdit_textEdited(const QString &arg1);

    void on_pushButton_3_clicked();

    void on_lineEdit_2_textEdited(const QString &arg1);

    void on_pushButton_5_clicked();

    void on_lineEdit_5_textEdited(const QString &arg1);

    void on_check_pattern();

    void on_pushButtonGo_clicked();

private:
    Ui::MainWindow *ui;
    string network_savefile_name="",data_set_file_name="";
    int radio_button_option;
    int no_of_threads;
    bool dataset_selected=false,network_file_selected=false;
    bool thread_manually_selected=true;
    float data_division=1;
    bool wait_a_little1=false;

    data_file_reader_class obj1;

    bool is_number(const std::string& s);
    bool is_float(const std::string& s);
    bool has_digits(string str);
    void message_display_system();
    bool valid_int_entry(QString);
};

#endif // MAINWINDOW_H
