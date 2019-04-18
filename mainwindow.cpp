#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include<QMessageBox>
#include<QDir>
#include<QScrollBar>
#include<qdebug.h>
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    pds_thread = new progress_display_thread(this);//thread creation
    connect(pds_thread,SIGNAL(progress_display_system_thread(QString)),this,SLOT(progress_display_system(QString)));//thread connector
    connect(pds_thread,SIGNAL(display_thread_options(bool)),this,SLOT(on_display_thread_options(bool)));
    connect(pds_thread,SIGNAL(task_complete_signal()),this,SLOT(on_task_complete_signal()));
    connect(pds_thread,SIGNAL(update_progress_bar()),this,SLOT(on_update_progress_bar()));
    connect(pds_thread,SIGNAL(set_label()),this,SLOT(on_set_label()));

    c_thread = new core_thread(this);

    idpc_thread = new input_data_pattern_checker_thread(this);
    connect(idpc_thread,SIGNAL(check_pattern()),this,SLOT(on_check_pattern()));

    ui->setupUi(this);
    //this->setFixedWidth(500);
    //this->setFixedHeight(500);
    //ui->textEdit->hide();
    //network file chooser
    ui->pushButton_1->hide();
    ui->label_2->hide();
    ui->lineEdit_4->hide();
    //dataset file chooser
    ui->pushButton_2->hide();
    ui->label_3->hide();
    ui->lineEdit_3->hide();
    ui->lineEdit->hide();
    ui->label_4->hide();
    //set threads
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->pushButton_3->hide();
    //progressbar
    ui->progressBar->hide();
    //start button
    ui->pushButton_4->hide();
    //ststus box
    ui->textEdit_3->setDisabled(true);
    ui->textEdit_3->setReadOnly(true);
    //option radio button 3 option
    ui->pushButton_5->hide();
    ui->lineEdit_5->hide();
    //stop button
    ui->lineEdit_5->hide();
    ui->pushButton_4->hide();
    ui->pushButtonGo->hide();
    ui->label_7->hide();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    QApplication::quit();
}

void MainWindow::on_check_pattern()
{
    int input_size=obj1.core1.body_engine_communication_data_obj.size_of_input_data;
    vector<float> input_data;
    input_data.clear();

    string str_input;
    str_input.clear();

    str_input.append(ui->lineEdit_5->text().toStdString());
    string number;
    number.clear();
    for(int a=0;a<str_input.length();a++)
    {
        string num(1,str_input.at(a));
        while(str_input.length()>a && is_number(num)==true)
        {
            string s(1,str_input.at(a));
            if(is_number(s)==true)
            {
                number.append(s);
                a++;
            }
            else
            {   break;}
        }
        if(a>=str_input.length())
        {   break;}
        string checker(1, str_input.at(a));
        if(is_number(checker)==false)
        {
            if(is_float(number))
            {   input_data.push_back(stoi(number));}
            number.clear();
        }
    }
    if(input_data.size()==input_size)
    {
        ui->pushButtonGo->setDisabled(false);
        obj1.core1.body_engine_communication_data_obj.input_data=input_data;
    }
    else
    {
        ui->pushButtonGo->setDisabled(true);
        obj1.core1.body_engine_communication_data_obj.input_data.clear();
    }

    Sleep(500);
    idpc_thread->pause=true;
}

void MainWindow::progress_display_system(QString t)//ontest
{
     ui->textEdit_3->setText(t);
     ui->textEdit_3->verticalScrollBar()->setValue(ui->textEdit_3->verticalScrollBar()->maximum());
}

void MainWindow::on_set_label()
{
    string strl7;
    strl7.clear();
    strl7="enter the "+to_string(obj1.core1.body_engine_communication_data_obj.size_of_input_data)+ " digit input in this format:- ";
    for(int a=0;a<obj1.core1.body_engine_communication_data_obj.size_of_input_data;a++)
    {
        strl7.append("a"+to_string(a));
        strl7.append(",");
    }
    QString qstr;
    qstr.clear();
    qstr.append(strl7.c_str());
    ui->label_7->setText(qstr);
    ui->label_7->show();
}

void MainWindow::on_display_thread_options(bool status)
{
    if(radio_button_option==0)
    {
        ui->label_5->setText("3. Do you want to set the no of threads manually?");
        ui->label_5->setStyleSheet("color:#0077dd;font-size: 17px;");
    }
    else if(radio_button_option==1)
    {
        ui->label_5->setText("4. Do you want to set the no of threads manually?");
        ui->label_5->setStyleSheet("color:#0077dd;font-size: 17px;");
    }
    if(status==true)
    {
        ui->label_5->show();
        ui->radioButton_5->show();
        //ui->radioButton_5->click();
        ui->radioButton_5->setDisabled(false);
        ui->radioButton_6->show();
        ui->radioButton_6->click();
        ui->label_6->show();
        ui->radioButton_6->setDisabled(false);
        ui->lineEdit_2->show();
        ui->lineEdit_2->setDisabled(true);
        ui->pushButton_3->show();
        ui->pushButton_3->setDisabled(false);
    }
    else
    {
        ui->label_5->hide();
        ui->radioButton_5->hide();
        ui->radioButton_6->hide();
        ui->label_6->hide();
        ui->lineEdit_2->hide();
        ui->pushButton_3->hide();
    }
}

void MainWindow::on_update_progress_bar()
{
    int datapacks_complete,total_c_datapacks;
    int predict_numerator,predict_denominator;
    ui->pushButton_3->setDisabled(true);
    obj1.core1.get_shared_block_data(datapacks_complete,total_c_datapacks,predict_numerator,predict_denominator);
    if(radio_button_option==0 || radio_button_option==1)
    {
        float done=datapacks_complete,total=total_c_datapacks;
        float value=(done/total)*100.0;
        if(total>0)
        {   ui->progressBar->setValue(value);}
        if(value>90 && wait_a_little1==false)
        {   obj1.core1.body_engine_communication_data_obj.add_message("\nwait a little bit longer...........");wait_a_little1=true;}
    }
    else if(radio_button_option==2)
    {
        float numerator=predict_numerator,denominator=predict_denominator;
        float value=(numerator/denominator)*100.0;
        if(denominator>0)
        {   ui->progressBar->setValue(value);}
    }
}

void MainWindow::on_task_complete_signal()
{
    //reset the variables
    network_savefile_name="";
    data_set_file_name="";
    dataset_selected=false;
    network_file_selected=false;
    thread_manually_selected=true;
    data_division=1;
    wait_a_little1=false;
    //ui changes ending
    ui->pushButton_4->setDisabled(false);
    ui->radioButton_1->setDisabled(false);
    ui->radioButton_2->setDisabled(false);
    ui->radioButton_3->setDisabled(false);
    ui->radioButton_4->setDisabled(false);
    //network file chooser
    ui->pushButton_1->setDisabled(false);
    ui->lineEdit_4->setDisabled(false);
    ui->lineEdit_4->clear();
    //dataset file chooser
    ui->pushButton_2->setDisabled(false);
    ui->lineEdit_3->setDisabled(false);
    ui->lineEdit_3->clear();
    //thread selection option
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->lineEdit_2->clear();
    ui->pushButton_3->hide();
    //data vivision
    ui->lineEdit->clear();
    ui->progressBar->hide();
    obj1.core1.restore_body_engine_communication_data_obj();//i dont think this option is needed anymore checking is required.
    obj1.reinitialize_the_core();
}

void MainWindow::on_radioButton_1_clicked()
{
    //network file chooser
    radio_button_option=0;
    dataset_selected=false;
    network_file_selected=false;
    ui->pushButton_1->hide();
    ui->label_2->hide();
    ui->lineEdit_4->hide();
    //dataset file chooser
    ui->pushButton_2->show();
    ui->label_3->show();
    ui->lineEdit_3->show();
    ui->lineEdit_3->clear();
    ui->label_3->setText("2. Select dataset file...");
    ui->label_3->setStyleSheet("color:#0077dd;font-size: 17px;");
    //data division
    ui->lineEdit->hide();
    ui->label_4->hide();
    //set threads
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->pushButton_3->hide();
    //start button
    ui->pushButton_4->show();
    ui->pushButton_4->setDisabled(true);
    //stop button
    ui->lineEdit_5->hide();
    ui->pushButton_5->hide();
}

void MainWindow::on_radioButton_2_clicked()
{
    //network file chooser
    radio_button_option=1;
    dataset_selected=false;
    network_file_selected=false;
    ui->pushButton_1->hide();
    ui->label_2->hide();
    ui->lineEdit_4->hide();
    ui->lineEdit_4->clear();
    //ui->label_2->setText("2. Select network file...");
    //ui->label_2->setStyleSheet("color:#0077dd;font-size: 17px;");
    //dataset file chooser
    ui->pushButton_2->show();
    ui->label_3->show();
    ui->lineEdit_3->show();
    ui->lineEdit_3->clear();
    ui->label_3->setText("2. Select dataset file...");
    ui->label_3->setStyleSheet("color:#0077dd;font-size: 17px;");
    //data division
    ui->lineEdit->show();
    ui->label_4->show();
    ui->label_4->setText("3. Data Division= ");
    ui->label_4->setStyleSheet("color:#0077dd;font-size: 17px;");
    ui->lineEdit->setDisabled(true);
    //set threads
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->pushButton_3->hide();
    //start button
    ui->pushButton_4->show();
    ui->pushButton_4->setDisabled(true);
    //stop button
    ui->lineEdit_5->hide();
    ui->pushButton_5->hide();
}

void MainWindow::on_radioButton_3_clicked()
{
    //network file chooser
    radio_button_option=2;
    dataset_selected=false;
    network_file_selected=false;
    ui->pushButton_1->show();
    ui->label_2->show();
    ui->lineEdit_4->show();
    ui->lineEdit_4->clear();
    ui->label_2->setText("2. Select network file...");
    ui->label_2->setStyleSheet("color:#0077dd;font-size: 17px;");
    //dataset file chooser
    ui->pushButton_2->show();
    ui->label_3->show();
    ui->lineEdit_3->show();
    ui->lineEdit_3->clear();
    ui->label_3->setText("3. Select dataset file...");
    ui->label_3->setStyleSheet("color:#0077dd;font-size: 17px;");
    //data division
    ui->lineEdit->hide();
    ui->label_4->hide();
    //set threads
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->pushButton_3->hide();
    //start button
    ui->pushButton_4->show();
    ui->pushButton_4->setDisabled(true);
    //stop button
    ui->lineEdit_5->hide();
    ui->pushButton_5->hide();
}

void MainWindow::on_radioButton_4_clicked()
{
    //network file chooser
    radio_button_option=3;
    dataset_selected=false;
    network_file_selected=false;
    ui->pushButton_1->show();
    ui->label_2->show();
    ui->lineEdit_4->show();
    ui->lineEdit_4->clear();
    ui->label_2->setText("3. Select network file...");
    ui->label_2->setStyleSheet("color:#0077dd;font-size: 17px;");
    //dataset file chooser
    ui->pushButton_2->hide();
    ui->label_3->hide();
    ui->lineEdit_3->hide();
    //data division
    ui->lineEdit->hide();
    ui->label_4->hide();
    //set threads
    ui->label_5->hide();
    ui->radioButton_5->hide();
    ui->radioButton_6->hide();
    ui->label_6->hide();
    ui->lineEdit_2->hide();
    ui->pushButton_3->hide();
    //start button
    ui->pushButton_4->show();
    ui->pushButton_4->setDisabled(true);
    //stop button
    ui->pushButton_5->hide();
    ui->label_7->hide();
}

void MainWindow::on_radioButton_5_clicked()
{
    ui->label_6->show();
    ui->lineEdit_2->show();
    ui->pushButton_3->show();
    ui->pushButton_3->setDisabled(true);
    ui->lineEdit_2->setDisabled(false);
    thread_manually_selected=true;
}

void MainWindow::on_radioButton_6_clicked()
{
    //ui->label_6->hide();
    ui->lineEdit_2->setDisabled(true);
    ui->pushButton_3->setDisabled(false);
    thread_manually_selected=false;
}

void MainWindow::on_pushButton_4_clicked()
{
    ui->pushButton_4->setDisabled(true);
    //main options
    ui->radioButton_1->setDisabled(true);
    ui->radioButton_2->setDisabled(true);
    ui->radioButton_3->setDisabled(true);
    ui->radioButton_4->setDisabled(true);
    //network file chooser
    ui->pushButton_1->setDisabled(true);
    ui->lineEdit_4->setDisabled(true);
    ui->label_2->setText("2. Select network file...");
    ui->label_2->setStyleSheet("color:#0077dd;font-size: 17px;");
    //dataset file chooser
    ui->pushButton_2->setDisabled(true);
    ui->lineEdit_3->setDisabled(true);
    if(radio_button_option==1)
    {   ui->label_3->setText("2. Select dataset file...");}
    else if(radio_button_option==2)
    {   ui->label_3->setText("3. Select dataset file...");}
    ui->label_3->setStyleSheet("color:#0077dd;font-size: 17px;");
    //data division
    ui->lineEdit->setDisabled(true);
    ui->label_4->setDisabled(true);
    //message display
    ui->textEdit_3->setDisabled(false);
    //rest modes
    if(radio_button_option==0 || radio_button_option==1 || radio_button_option==2)
    {
        //data_file_reader_class *obj=new data_file_reader_class();        
        obj1.core1.body_engine_communication_data_obj.task_complete=false;
        pds_thread->set_object(&obj1);
        c_thread->set_object_and_data(&obj1,data_set_file_name,network_savefile_name,radio_button_option,data_division);

        //obj1.core_starter(data_set_file_name,radio_button_option,data_division,network_savefile_name);
        c_thread->start();
        pds_thread->start();
        //obj1.core1.body_engine_communication_data_obj.task_complete=true;
    }
    //progress bar predict mode only
    if(radio_button_option==2)
    {
        obj1.core1.body_engine_communication_data_obj.progress_bar_display=true;
        ui->progressBar->show();
        ui->progressBar->setValue(0);
    }
    else if(radio_button_option==3)
    {
        //ui changes
        ui->pushButton_4->hide();//start
        ui->pushButton_5->show();//stop
        ui->lineEdit_5->show();
        ui->pushButtonGo->show();
        ui->pushButtonGo->setDisabled(true);
        //start the engine
        obj1.core1.body_engine_communication_data_obj.task_complete=false;
        pds_thread->set_object(&obj1);
        c_thread->set_object_and_data(&obj1,data_set_file_name,network_savefile_name,radio_button_option,data_division);
        c_thread->start();
        pds_thread->start();

        idpc_thread->set_object(&obj1);
        idpc_thread->start();
    }
}

void MainWindow::on_pushButton_1_clicked()
{
    network_file_selected=false;
    QString selfilter = tr("csv (*.csv)");
    QString network_file_path("");
    network_file_path=QFileDialog::getOpenFileName(this,"Select a network save file...",QDir::currentPath(),tr("All files (*.*);;csv (*.csv)" ),&selfilter );
    QFileInfo file(network_file_path);
    ui->lineEdit_4->setText(network_file_path);
    network_savefile_name.clear();
    network_savefile_name.append(file.completeBaseName().toStdString());
    network_savefile_name.append(".");
    network_savefile_name.append(file.completeSuffix().toStdString());
    ui->pushButton_4->setDisabled(true);
    if(radio_button_option==2 && network_file_path.length()>0 && dataset_selected==true)
    {   ui->pushButton_4->setDisabled(false);}
    else if(radio_button_option==3 && network_file_path.length()>0)
    {   ui->pushButton_4->setDisabled(false);}
    if(network_file_path.length()>0)
    {   network_file_selected=true;}
    network_file_path.clear();
}

void MainWindow::on_pushButton_2_clicked()
{
    dataset_selected=false;
    QString selfilter = tr("csv (*.csv)");
    QString dataset("");
    dataset=QFileDialog::getOpenFileName(this,"Select a dataset file...",QDir::currentPath(),tr("All files (*.*);;csv (*.csv)" ),&selfilter );
    QFileInfo file(dataset);
    ui->lineEdit_3->setText(dataset);
    data_set_file_name.clear();
    data_set_file_name.append(file.completeBaseName().toStdString());
    data_set_file_name.append(".");
    data_set_file_name.append(file.completeSuffix().toStdString());
    ui->pushButton_4->setDisabled(true);
    if(radio_button_option==0 && dataset.length()>0)
    {   ui->pushButton_4->setDisabled(false);}
    else if(radio_button_option==2 && dataset.length()>0 && network_file_selected==true)
    {   ui->pushButton_4->setDisabled(false);}
    if(radio_button_option==1 && dataset.length()>0)
    {   ui->lineEdit->setDisabled(false);}
    if(dataset.length()>0)
    {   dataset_selected=true;}
    dataset.clear();
}

bool MainWindow::is_number(const string& s)//working
{   return !s.empty() && find_if(s.begin(),s.end(), [](char c) { return !isdigit(c); }) == s.end();}

bool MainWindow::is_float(const string &s)//working
{
    try{
            stof(s);
            return true;
    }
    catch(...){
        return false;
    }
}

bool MainWindow::has_digits(string str)
{
    unsigned int i=str.at(str.length()-1);
    if(i>=48 && i<=57)
    {   return true;}
    else
    {   return false;}
}

void MainWindow::on_lineEdit_textEdited(const QString &arg1)
{
    ui->pushButton_4->setDisabled(true);
    if((is_float(arg1.toStdString())==true && dataset_selected==true && radio_button_option==1))
    {
        if(has_digits((arg1.toStdString()))==true)
        {
            //cout<<stof(arg1.toStdString())<<endl;
            if(stof(arg1.toStdString())!=1 && stof(arg1.toStdString())!=0)
            {
                data_division=stof(arg1.toStdString());
                ui->pushButton_4->setDisabled(false);
            }
        }
    }
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->label_5->setDisabled(true);
    ui->radioButton_5->setDisabled(true);
    ui->radioButton_6->setDisabled(true);
    ui->lineEdit_2->setDisabled(true);
    ui->pushButton_3->setDisabled(true);
    if(thread_manually_selected==true)
    {
        obj1.core1.body_engine_communication_data_obj.thread_set_automatic=false;
        obj1.core1.body_engine_communication_data_obj.set_no_of_threads=no_of_threads;
    }
    else
    {   obj1.core1.body_engine_communication_data_obj.thread_set_automatic=true;}
    //clutch relesed on the engine
    obj1.core1.body_engine_communication_data_obj.no_of_threads_setting_mode=true;
    //progress par for option 0 and 1
    if(radio_button_option==0 || radio_button_option==1)
    {   obj1.core1.body_engine_communication_data_obj.progress_bar_display=true;
        ui->progressBar->show();
        ui->progressBar->setValue(0);
    }
}

bool MainWindow::valid_int_entry(QString str)
{
    ui->pushButton_3->setDisabled(true);
    try{
        if(is_number(str.toStdString())==true)
        {
            if(stoi(str.toStdString())!=0)
            {
                if(has_digits((str.toStdString()))==true)
                {
                    no_of_threads=stoi(str.toStdString());
                    //cout<<no_of_threads<<endl;
                    ui->pushButton_3->setDisabled(false);
                }
            }
        }
        return true;
    }
    catch(...){
        return false;
    }

}

void MainWindow::on_lineEdit_2_textEdited(const QString &arg1)
{   valid_int_entry(arg1);}

void MainWindow::on_pushButton_5_clicked()
{
    obj1.core1.body_engine_communication_data_obj.task_complete=true;//stop the engine first and than reinitialize it for further use.
    Sleep(500);
    on_task_complete_signal();

    ui->pushButton_4->show();
    ui->pushButton_4->setDisabled(true);
    ui->pushButton_5->hide();
    ui->lineEdit_5->hide();
    ui->label_7->hide();
    ui->pushButtonGo->hide();
    ui->lineEdit_5->clear();
}

void MainWindow::on_lineEdit_5_textEdited(const QString &arg1)
{   idpc_thread->pause=false;}

void MainWindow::on_pushButtonGo_clicked()
{   obj1.core1.body_engine_communication_data_obj.data_entered=true;}
