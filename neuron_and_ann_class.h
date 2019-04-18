/*
Neuron class 
ANN class
*/

#include<iostream>
#include<vector>
#include<string>
#include<string.h>
#include<math.h>
#include<pthread.h>

using namespace std;

int firing_point=90; //30 40

pthread_mutex_t lock;

class neuron{
    private:
    int id;
    bool input_id,output_id;
    bool fire_status=false;
    float data,fp;
    public:
    
    void set_neuron_identity(int i,bool i_id,bool o_id)
    {
        id=i;
        input_id=i_id;
        output_id=o_id;
    }

    void set_data(float d)
    {
        data=d;
        //cout<<"\ndata="<<d<<endl;
        if(d>firing_point){//i had to change it because of the private solver to -30
            fire_status=true;
        }
        fp=firing_point;
    }

    void change_fire_status(bool s)
    {   fire_status=s;}

    void reset_fire_status()
    {   fire_status=false;}

    bool return_fire_status()
    {   return fire_status;}
    
    int get_id()
    {   return id;}
    
    float return_data()
    {   return data;}

    float return_firing_point()
    {   return fp;}
};

struct path_struct{
    vector<int> input_neuron_id;
    int output_neuron_id,path_id;
    vector<float> weight_matrix;
};

class ann{
    private:
    float accuracy=0;
    int label_neuron_to_be_fired_id=0;
    bool label_neuron_reset_status=false;
   
    float correct=0;
    int total=0;
    
    public:
    vector<float> elements;
    void set_elements_vector(vector<float> e)
    {   elements=e;}

    float return_label_for_firing_neuron_index(int index)
    {   return elements[index];}
    
    void enter_label_neuron_to_be_fired(int label,vector<float> id){
        label_neuron_reset_status=true;
        for(int a=0;a<id.size();a++)
        {
            if(id[a]==label)
            {   label_neuron_to_be_fired_id=a;}
            output_neurons[a].reset_fire_status();
        }
    }    

    vector<neuron> input_neurons;
    vector<neuron> output_neurons;
    vector<float> mean_buffer;//what the fuck is this????
    vector<path_struct> path;
    bool predict_mode=false;

    void reset_all_output_neurons()
    {
        label_neuron_reset_status=true;
        for(int a=0;a<output_neurons.size();a++)
        {   output_neurons[a].reset_fire_status();}
    }

    int return_path_size()
    {   return path.size();}
    
    void enter_data_in_the_network(vector<float> input_matrix) //data is pushed into the input neurons.
    {
        for(int a=0;a<input_neurons.size();a++)
        {   input_neurons[a].set_data(input_matrix[a]);}
    }

    void set_no_of_input_neuron(int n)
    {   
        input_neurons.resize(n);
        for(int a=0;a<input_neurons.size();a++)
        {   input_neurons[a].set_neuron_identity(a,true,false);}
    }

    void set_no_of_output_neuron(int n)
    {   
        output_neurons.resize(n);
        for(int a=0;a<output_neurons.size();a++)
        {   output_neurons[a].set_neuron_identity(a,false,true);}
    }

    void create_new_path(vector<float> weight_matrix,int output_id)
    {
        pthread_mutex_lock(&lock);
        
        path_struct new_path;
        int id=path.size();
        if(id==0)
        {   new_path.path_id=0;}
        else
        {
            id=path[id-1].path_id;
            id++;
            new_path.path_id=id;
        }
        //input neuron ids are not required but still included for would be future requeriments
        for(int a=0;a<input_neurons.size();a++)
        {new_path.input_neuron_id.push_back(input_neurons[a].get_id());}
        new_path.output_neuron_id=output_id;
        new_path.weight_matrix.clear();
        for(int a=0;a<weight_matrix.size();a++)
        {   new_path.weight_matrix.push_back(weight_matrix[a]);}

        path.push_back(new_path);
        
        pthread_mutex_unlock(&lock);
    }

    int input_neuron_size()
    {   return input_neurons.size();}

    int output_neuron_size()
    {   return output_neurons.size();}

    bool check_network_validity()
    {
        if(input_neuron_size()==0||output_neuron_size()==0)
        {   return false;}
        else
        {   return true;}
    }

    float sigmoid(float summation)
    {
        float exp_value=exp((double)-summation);
        float output=1/(1+exp_value);
        return output;
    }

    void propagate(){
        float summation=0;
        vector<float> summation_vec;
        vector<int> total_no_of_fires_of_paths;
        total_no_of_fires_of_paths.clear();
        bool sure_positive_found=false,second_neuron_fired_status=false;
        int check1=0;
        for(int a=0;a<output_neuron_size();a++)
        {
            summation=0;
            summation_vec.clear();
            firing_point=0;
            for(int c=0;c<path.size();c++)
            {
                float summation_temp=0;
                if(path[c].output_neuron_id==a)
                {
                    firing_point+=50;
                    for(int d=0;d<path[c].weight_matrix.size();d++)//weight matrix size = input neuron size
                    {
                        summation_temp=summation_temp+input_neurons[d].return_data()*path[c].weight_matrix[d]; //need to be modified...................................
                        //cout<<"i= "<<input_neurons[d].return_data()<<" w= "<<path[c].weight_matrix[d]<<endl;
                    }
                    if(summation_temp>500 || summation_temp<-500)//1000
                    {
                        summation_temp=0;
                        firing_point-=50;
                    }
                    else if(summation_temp>0&&summation_temp<40)
                    {
                        summation_temp=0;
                        firing_point-=50;
                    }
                    //cout<<"\nsummation_temp1= "<<summation_temp;
                    summation_temp=((atan(summation_temp)*180/3.1415)/90)*100;
                    summation_vec.push_back(summation_temp);
                    //cout<<"\nsummation_temp2= "<<summation_temp;
                    summation=summation+summation_temp;
                }
            }
            if(summation-firing_point>100000)
            {   summation=0;}
            if(summation>=100000)
            {   summation=0;}
            //cout<<"\nsummation= "<<summation<<" label_neuron_to_be_fired_id= "<<label_neuron_to_be_fired_id<<"a= "<<a;
            //cout<<" firing_point= "<<firing_point;
            output_neurons[a].set_data(summation);
            //cout<<"\ncheck!!!!!";
            //int gh;cin>>gh;
            //checker for df.
            if(output_neurons[a].return_fire_status()==true)
            {   check1++;}
            if(a==output_neuron_size()-1 && check1>1)
            {
                vector<neuron*> temp_neuron_vect;
                neuron *temp_neuron;
                temp_neuron_vect.clear();
                for(int b=0;b<output_neuron_size();b++)
                {
                    //cout<<"\nsummation= "<<output_neurons[b].return_data()<<" label_neuron_to_be_fired_id= "<<label_neuron_to_be_fired_id<<"b= "<<b;
                    //cout<<" firing_point= "<<output_neurons[b].return_firing_point();
                    if(output_neurons[b].return_fire_status()==true)
                    {
                        temp_neuron_vect.push_back(&output_neurons[b]);
                    }
                }
                //sorting
                for(int b=0;b<temp_neuron_vect.size();b++)
                {
                    for(int c=0;c<temp_neuron_vect.size();c++)
                    {
                        if((temp_neuron_vect[b]->return_data()-temp_neuron_vect[b]->return_firing_point()) > (temp_neuron_vect[c]->return_data()-temp_neuron_vect[c]->return_firing_point()))
                        {
                            temp_neuron=temp_neuron_vect[b];
                            temp_neuron_vect[b]=temp_neuron_vect[c];
                            temp_neuron_vect[c]=temp_neuron;
                        }
                    }
                }
                for(int b=0;b<temp_neuron_vect.size();b++)
                {
                    //cout<<"\nsummation= "<<temp_neuron_vect[b]->return_data();
                    //cout<<" firing_point= "<<temp_neuron_vect[b]->return_firing_point();
                    if(b==0)
                    {   temp_neuron_vect[b]->change_fire_status(true);}
                    else
                    {   temp_neuron_vect[b]->change_fire_status(false);}
                }
            }
            //democracy module 
            /*int votes=0;
            for(int b=0;b<summation_vec.size();b++)
            {
                if(summation_vec[a]>70)
                {
                    votes++;
                }
            }
            if(votes>=summation_vec.size()/2)
            {   
                output_neurons[a].set_data(summation);
                output_neurons[a].change_fire_status(true);
            }
            else
            {
                output_neurons[a].set_data(summation);
                output_neurons[a].change_fire_status(false);
            }*/
        }
        //nf analizer
        bool one_fired=false;
        for(int a=0;a<output_neuron_size();a++)
        {
            if(output_neurons[a].return_fire_status()==true)
            {
                one_fired=true;
            }
        }
        if(one_fired==false)
        {
            /*cout<<"\nneuron to fire= "<<label_neuron_to_be_fired_id;
            for(int a=0;a<output_neurons.size();a++)
            {
                cout<<"\na= "<<a<<" neuron_data= "<<output_neurons[a].return_data()<<" firing_point= "<<output_neurons[a].return_firing_point();
            }
            cout<<"\ncheck!!!!!";
            int gh;cin>>gh;*/
            vector<neuron*> neuron_temp_vect;
            neuron *neuron_temp;
            neuron_temp_vect.clear();
            for(int a=0;a<output_neuron_size();a++)
            {   neuron_temp_vect.push_back(&output_neurons[a]);}
            //sorting
            for(int a=0;a<output_neuron_size();a++)
            {
                for(int b=0;b<output_neuron_size();b++)
                {
                    if(neuron_temp_vect[a]->return_data()>neuron_temp_vect[b]->return_data())
                    {
                        neuron_temp=neuron_temp_vect[a];
                        neuron_temp_vect[a]=neuron_temp_vect[b];
                        neuron_temp_vect[b]=neuron_temp;
                    }
                }
            }
            for(int a=0;a<neuron_temp_vect.size();a++)
            {
                if(neuron_temp_vect[a]->return_firing_point()<100000)
                {   neuron_temp_vect[a]->change_fire_status(true);break;}
            }
        }
        if(predict_mode==false)
        {   accuracy_calc(label_neuron_to_be_fired_id);}
    }
    int double_fire=0,all_not_fired=0,wrongly_fired=0;
    void accuracy_calc(int label_neuron_id){
        int status=0;
        label_neuron_reset_status=false;
        total++;
        if(output_neurons[label_neuron_id].return_fire_status()==true)
        {
            for(int a=0;a<output_neuron_size();a++)//for checking multiple fire status.
            {
                if(a==label_neuron_id)
                {   continue;}
                if(output_neurons[a].return_fire_status()==true)
                {
                    status++;
                }
            }
            if(status==0)
            {
                correct++;
            }
            else
            {   double_fire++;}
        }
        else{
            int count=0;
            for(int a=0;a<output_neurons.size();a++)
            {
                if(output_neurons[a].return_fire_status()==false)
                {   count++;}
            }
            if(count==output_neuron_size())
            {   all_not_fired++;}
            else
            {   wrongly_fired++;}
        }
        accuracy=(correct/total)*100;
    }

    float return_accuracy()
    {   cout<<"\ncorrect= "<<correct<<" incorrect= "<<total-correct<<" total= "<<total<<" double_fire= "<<double_fire<<" all_not_fired= "<<all_not_fired<<" wrongly_fired= "<<wrongly_fired<<endl;
        return accuracy;
    }

    void reset_statistics()
    {
        correct=0;
        total=0;
        double_fire=0;
        all_not_fired=0;
        wrongly_fired=0;
        accuracy=0;
    }
};