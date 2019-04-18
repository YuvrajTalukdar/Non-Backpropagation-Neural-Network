/*
core class
*/
//#define _GNU_SOURCE
#include<iostream>
#include<vector>
#include<fstream>
#include<sys/stat.h>
#include<string>
#include<string.h>
#include<algorithm>
#include<math.h>
#include<stdlib.h>
#include<time.h>
//#include<pthread.h>//for linux threads
#include<thread>
#include<windows.h>
#include<algorithm>//for strcasestr
#include<functional>//for strcasestr
#include<cctype>//for strcasestr

#include<QDirIterator>
#include<QDir>
#include<qdebug.h>

#include"neuron_and_ann_class.h"
#include"data_package_class.h"//should be removed

using namespace std;

static bool display_iterations=false;//iteration display switch for debugging the code
//static bool pds=true;//progress_display_system for displaying the progress bar
//int fg=2;//for the checker
struct shared_block_data//required for 
{
    int predict_progress_bar_numerator=0;//required for prediction progress bar
    int predict_progress_bar_denominator=0;//required for prediction progress bar
    int no_of_c_datapacks_completed=0;//required for training progress bar
    int total_c_datapacks;//required for training progress bar

};
inline shared_block_data shared_block_data_obj;//lnline for migw 64 bit and static for mingw 32 bit. The 32 bit version dont support inline feature for c++17 for the progress bar dont work if compiled for 32 bit

struct datapack_structure_defination{
    int no_of_labels;
    int no_of_elements_in_each_record;
    vector<float> elements;
};

struct converted_data_pack{
    vector<vector<float>> firing_data;
    vector<vector<float>> not_firing_data;
    vector<float> objective_function_coefficients;
    vector<float> weight_matrix;//ans stored here.
    int firing_neuron_index;
    float firing_label;
    float lower_firing_constrain_rhs=55; //75,70,68,55

    float upper_not_firing_constrain_rhs=3; //2,3
};

struct conflicting_data_id1
{
    bool conflict_id_present=false;
    bool mother_fucking_bool[7];//this motherfucker is required to satisfy the 64 bit compiler
    vector<int> id;
    //bool conflicting_id_present=false;
};

class ratio_locker
{
    private:
    bool lock_enabled=false;
    int no_of_labels;
    vector<int> no_of_data_for_each_labels;
    vector<float> ratios;
    public:
    void set_ratios(vector<int> data_no_vec)
    {
        if(lock_enabled==false)
        {
            lock_enabled=true;
            no_of_data_for_each_labels=data_no_vec;
            no_of_labels=data_no_vec.size();
            for(int a=0;a<no_of_labels;a++)
            {
                float sum=0;
                for(int b=0;b<no_of_labels;b++)
                {
                    if(b!=a)
                    {
                        sum=sum+data_no_vec[b];
                    }
                }
                ratios.push_back(data_no_vec[a]/sum);
            }
        }
    }

    void un_lock()
    {   lock_enabled=false;}

    void lock()
    {   lock_enabled=true;}

    float get_ratio(int label_index)
    {   return ratios[label_index];}

    float get_default_ratio()
    {   return 1;}

    bool is_locked()
    {   return lock_enabled;}
};
static ratio_locker ratio_locker1;

class modified_simplex_solver{
    private:
    struct id{
            bool slack=false,basic=false,z=false,rhs=false,theta=false;
            int id;
        };

    struct simplex_table{

        vector<id> c_id;//no_of_columns-rhs-z
        vector<id> r_id; //no_of_rows-z_row
        vector<vector<float>> basic_var; //no_of_column-slack_var-z-rhs-theta*no_of_rows-z_row
        vector<vector<float>> slack_var; //no_of_columns-basic_var-rhs-theta*no_of_rows-z_row
        vector<float> z_col;//no_of_rows-z_row
        vector<double> rhs;//no_of_rows-z_row
        vector<long double> theta;//no_of_rows-z_row
        vector<float> z_row;//no_of_columns
    };

    class make_solution_feasible{

        private:
        bool termination_condition_checker(simplex_table* st)
        {
            bool status=true;
            for(int a=0;a<st->r_id.size();a++)
            {
                if(st->r_id[a].slack==true)
                {
                    if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>=0)
                    {
                        status=false;
                        break;
                    }
                }
                if(status==false)
                {   break;}
            }
            if(status==true)
            {
                //cout<<"\n\ntraining ended check table";
            }
            return status;
        }

        void display_st(simplex_table st)
        {
            fstream file1("simplex_table.csv",ios::out);
            file1<<",";
            for(int a=0;a<st.c_id.size();a++)
            {
                if(st.c_id[a].basic==true)
                {
                    file1<<"c"<<st.c_id[a].id<<",";
                }
                else if(st.c_id[a].slack==true)
                {
                    file1<<"s"<<st.c_id[a].id<<",";
                }
                else if(st.c_id[a].z==true)
                {
                    file1<<"z"<<",";
                }
                else if(st.c_id[a].rhs==true)
                {
                    file1<<"rhs"<<",";
                }
                else if(st.c_id[a].theta==true)
                {
                    file1<<"theta,";
                }
            }
            file1<<"\n";
            for(int a=0;a<st.r_id.size();a++)
            {
                if(st.r_id[a].basic==true)
                {
                    file1<<"c"<<st.r_id[a].id<<",";
                }
                else if(st.r_id[a].slack==true)
                {
                    file1<<"s"<<st.r_id[a].id<<",";
                }
                else if(st.r_id[a].z==true)
                {
                    file1<<"z"<<",";
                }
                else if(st.r_id[a].rhs==true)
                {
                    file1<<"rhs"<<",";
                }


                for(int c=0;c<st.basic_var[a].size();c++)
                {
                    file1<<st.basic_var[a][c]<<",";
                }
                for(int c=0;c<st.slack_var[a].size();c++)
                {
                    file1<<st.slack_var[a][c]<<",";
                }
                file1<<st.z_col[a]<<","<<st.rhs[a]<<","<<st.theta[a]<<",";

                file1<<"\n";
            }
            file1<<"z,";
            for(int a=0;a<st.z_row.size();a++)
            {
                file1<<st.z_row[a]<<",";
            }

            file1.close();
        }

        long double round_to_zero(long double input)
        {
            if(input<0.001 && input>-0.001)
            {   return input;}//0;}
            else
            {   return input;}
        }

        void simplex_table_modifier(int p_row_index,int p_col_index,simplex_table* st)
        {
            /*if(fg==1)
            {
                display_st(*st);
                int gh;cin>>gh;
            }*/
            
            //row id changer
            st->r_id[p_row_index].basic=st->c_id[p_col_index].basic;
            st->r_id[p_row_index].id=st->c_id[p_col_index].id;
            st->r_id[p_row_index].rhs=st->c_id[p_col_index].rhs;
            st->r_id[p_row_index].slack=st->c_id[p_col_index].slack;
            st->r_id[p_row_index].theta=st->c_id[p_col_index].theta;
            st->r_id[p_row_index].z=st->r_id[p_col_index].z;

            float pe;bool checker=false;
            //pivot element extractor
            if(p_col_index<st->basic_var[0].size())
            {
                pe=st->basic_var[p_row_index][p_col_index];
                if(display_iterations==true)
                {   cout<<"\n\ncheck table";}
            }
            else
            {
                int slack_p_col=p_col_index-st->basic_var[0].size();
                pe=st->slack_var[p_row_index][slack_p_col];
                if(display_iterations==true)
                {
                    cout<<"pe= "<<pe<<endl;
                    cout<<"\n\ncheck table";
                }
                checker=true;
            }

            //pivot row modifier
            for(int a=0;a<st->basic_var[p_row_index].size();a++)
            {   st->basic_var[p_row_index][a]=st->basic_var[p_row_index][a]/pe;}

            for(int a=0;a<st->slack_var[p_row_index].size();a++)
            {   st->slack_var[p_row_index][a]=st->slack_var[p_row_index][a]/pe;}

            st->z_col[p_row_index]=st->z_col[p_row_index]/pe;
            double rhs_temp1=st->rhs[p_row_index],pe_temp1=pe;
            st->rhs[p_row_index]=rhs_temp1/pe_temp1;

            //rest of the row - z_row modifier
            bool basic_point=true;
            int p_col_index_temp;
            if(p_col_index>=st->basic_var[0].size())
            {
                p_col_index_temp=p_col_index-st->basic_var[0].size();
                basic_point=false;
            }

            long double multiplying_element;
            for(int a=0;a<st->basic_var.size();a++)
            {
                //multiplying element finder
                bool p_row=false;
                if(basic_point==true)
                {
                    if(a==p_row_index)
                    {   continue;}
                    multiplying_element=st->basic_var[a][p_col_index];
                }
                else
                {
                    if(a==p_row_index)
                    {   continue;}
                    multiplying_element=st->slack_var[a][p_col_index_temp];
                }
                //basic_var row modification instruction
                for(int b=0;b<st->basic_var[a].size();b++)
                {
                    st->basic_var[a][b]=round_to_zero(st->basic_var[a][b]-multiplying_element*st->basic_var[p_row_index][b]);
                }
                //slack_var modification instruction
                for(int b=0;b<st->slack_var[a].size();b++)
                {
                    st->slack_var[a][b]=round_to_zero(st->slack_var[a][b]-multiplying_element*st->slack_var[p_row_index][b]);
                }

                //z col modification
                st->z_col[a]=round_to_zero(st->z_col[a]-multiplying_element*st->z_col[p_row_index]);
                //rhs col modification instruction
                st->rhs[a]=st->rhs[a]-multiplying_element*st->rhs[p_row_index];
            }

            //z row modifier
            if(display_iterations==true)
            {   cout<<"\np_col_index= "<<p_col_index<<" multiplying element= "<<multiplying_element;}
            multiplying_element=st->z_row[p_col_index];
            vector<float> basic_plus_slack_plus_z_plus_rhs_temp;
            basic_plus_slack_plus_z_plus_rhs_temp.clear();
            for(int b=0;b<st->basic_var[p_row_index].size();b++)
            {   basic_plus_slack_plus_z_plus_rhs_temp.push_back(round_to_zero(st->basic_var[p_row_index][b]));}
            for(int b=0;b<st->slack_var[p_row_index].size();b++)
            {   basic_plus_slack_plus_z_plus_rhs_temp.push_back(round_to_zero(st->slack_var[p_row_index][b]));}
            basic_plus_slack_plus_z_plus_rhs_temp.push_back(round_to_zero(st->z_col[p_row_index]));
            basic_plus_slack_plus_z_plus_rhs_temp.push_back(round_to_zero(st->rhs[p_row_index]));

            for(int b=0;b<st->z_row.size();b++)
            {
                //cout<<"\nz_row= "<<st->z_row[b]<<" basic_s_z_r_t= "<<basic_plus_slack_plus_z_plus_rhs_temp[b];
                st->z_row[b]=round_to_zero(st->z_row[b]-multiplying_element*basic_plus_slack_plus_z_plus_rhs_temp[b]);

            }
            //display_st(*st);

            //termination condition checker
            if(termination_condition_checker(st)==false)
            {   pivot_element_finder(st);}
            else
            {
                //solution found saving it

            }
        }

        conflicting_data_id1 conflict_id;

        void conflicting_data_finder(simplex_table* st)
        {
            conflict_id.conflict_id_present=true;
            conflict_id.id.clear();
            for(int a=0;a<st->r_id.size();a++)
            {
                if(st->r_id[a].slack==true)
                {
                    if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>0) //a bug may be present here.
                    {
                        conflict_id.id.push_back(a);
                    }
                }
            }
           //module working tested problem present in the conflictig data detection
        }

        int iteration=0;
        void pivot_element_finder(simplex_table* st)
        {
            iteration++;
            if(display_iterations==true)
            {   
                cout<<"\niteration= "<<iteration;
                cout<<"r_id size= "<<st->r_id.size();
            }
            //pivot column finder
            int p_col_index,p_row_index;
            int row_with_negative_slack;
                //row havaing negative slack finder
            bool p_col_found=false;
            int row_skip=1;
            bool row_with_negative_slack_found=false;
            for(int a=0;a<st->r_id.size();a++)
            {
                if(st->r_id[a].slack==true)
                {
                    if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>0) //a bug may be present here.
                    {
                        row_with_negative_slack=a;
                        row_with_negative_slack_found=true;
                        break;
                    }
                }
            }

            if(row_with_negative_slack_found==true)
            {
                if(display_iterations==true)
                {   cout<<"\nrow_with_negative_slack= "<<row_with_negative_slack;}
                    //pivot column finding process
                        //search in basic var vector

                for(int a=0;a<st->basic_var[row_with_negative_slack].size();a++)
                {
                    if(st->basic_var[row_with_negative_slack][a]>0)
                    {
                        p_col_index=a;
                        p_col_found=true;
                        break;
                    }
                }
                        //search in slack var vector
                if(p_col_found==false)
                {
                    for(int a=0;a<st->slack_var[row_with_negative_slack].size();a++)
                    {
                        if(st->slack_var[row_with_negative_slack][a]>0)
                        {
                            p_col_index=a+st->basic_var[row_with_negative_slack].size();
                            p_col_found=true;
                            break;
                        }
                    }
                }

                if(p_col_found==false)
                {
                    conflicting_data_finder(st);
                }
                else
                {
                    conflict_id.id.clear();
                    conflict_id.conflict_id_present=false;
                    st->theta.clear();
                    for(int a=0;a<st->r_id.size();a++)
                    {
                        if(p_col_index<st->basic_var[a].size())
                        {
                            //cout<<"\ntheta"<<st->rhs[a]/st->basic_var[a][p_col_index];
                            if(st->basic_var[a][p_col_index]==0)
                            {
                                st->theta.push_back(0);
                            }
                            else
                            {
                                long double rhs=st->rhs[a],basic_var=st->basic_var[a][p_col_index];//if long double is not used than precision problem may occur which may result in looping which even bland's rule fails to solve
                                long double calc=rhs/basic_var;//st->rhs[a]/st->basic_var[a][p_col_index];
                                st->theta.push_back(calc);
                            }
                        }
                        else
                        {
                            int temp_col_index=p_col_index-st->basic_var[a].size();
                            //cout<<"\np_col_index= "<<p_col_index<<" bas var size= "<<st->basic_var[a].size();
                            //cout<<"temp col index= "<<temp_col_index<<endl;
                            if(st->slack_var[a][temp_col_index]==0)
                            {
                                st->theta.push_back(0);
                            }
                            else
                            {
                                long double rhs=st->rhs[a],slack_var=st->slack_var[a][temp_col_index];
                                long double calc=rhs/slack_var;//st->rhs[a]/st->slack_var[a][temp_col_index];
                                st->theta.push_back(calc);
                            }
                        }
                    }
                    //pivot row finder
                    vector<long double> sorted_theta;
                    sorted_theta=st->theta;
                    sort(sorted_theta.begin(),sorted_theta.end());
                    long double smallest_positive_theta;
                    for(int a=0;a<sorted_theta.size();a++)
                    {
                        if(sorted_theta[a]>0)
                        {   smallest_positive_theta=sorted_theta[a];
                            break;
                        }
                    }
                    
                    for(int a=0;a<st->theta.size();a++)
                    {
                        if(st->theta[a]==smallest_positive_theta)
                        {
                            p_row_index=a;
                            break;
                        }
                    }
                    if(display_iterations==true)
                    {
                        cout<<"\nsmallest positive theta= "<<smallest_positive_theta;
                        cout<<"\npivot row= "<<p_row_index<<" pivot column= "<<p_col_index;
                    }
                    simplex_table_modifier(p_row_index,p_col_index,st);
                }
            }
        }

        public:
        conflicting_data_id1 return_conflict_id_pack()
        {
            return conflict_id;
        }

        void start(simplex_table* st)
        {   pivot_element_finder(st);}

    }feasible_solution_calculator;

    void display_st(simplex_table st)
    {
        fstream file1("simplex_table.csv",ios::out);
        file1<<",";
        for(int a=0;a<st.c_id.size();a++)
        {
            if(st.c_id[a].basic==true)
            {
                file1<<"c"<<st.c_id[a].id<<",";
            }
            else if(st.c_id[a].slack==true)
            {
                file1<<"s"<<st.c_id[a].id<<",";
            }
            else if(st.c_id[a].z==true)
            {
                file1<<"z"<<",";
            }
            else if(st.c_id[a].rhs==true)
            {
                file1<<"rhs"<<",";
            }
            else if(st.c_id[a].theta==true)
            {
                file1<<"theta,";
            }
        }
        file1<<"\n";
        for(int a=0;a<st.r_id.size();a++)
        {
            if(st.r_id[a].basic==true)
            {
                file1<<"c"<<st.r_id[a].id<<",";
            }
            else if(st.r_id[a].slack==true)
            {
                file1<<"s"<<st.r_id[a].id<<",";
            }
            else if(st.r_id[a].z==true)
            {
                file1<<"z"<<",";
            }
            else if(st.r_id[a].rhs==true)
            {
                file1<<"rhs"<<",";
            }


            for(int c=0;c<st.basic_var[a].size();c++)
            {
                file1<<st.basic_var[a][c]<<",";
            }
            for(int c=0;c<st.slack_var[a].size();c++)
            {
                file1<<st.slack_var[a][c]<<",";
            }
            file1<<st.z_col[a]<<","<<st.rhs[a]<<","<<st.theta[a]<<",";

            file1<<"\n";
        }
        file1<<"z,";
        for(int a=0;a<st.z_row.size();a++)
        {
            file1<<st.z_row[a]<<",";
        }

        file1.close();
    }

    public:
    void start_solver(converted_data_pack* cdp)
    {
        if(display_iterations==true)
        {   cout<<"\n\ncdp->firing_data.size()= "<<cdp->firing_data.size();}
        id temp_id;
        simplex_table st;
        st.c_id.clear();
        for(int a=0;a<cdp->firing_data[0].size()*2;a++)
        {
            temp_id.basic=true;
            temp_id.slack=false;
            temp_id.z=false;
            temp_id.rhs=false;
            temp_id.theta=false;
            temp_id.id=a;
            st.c_id.push_back(temp_id);
        }
        int slack_id=cdp->firing_data[0].size()*2;
        for(int a=cdp->firing_data[0].size();a<(cdp->firing_data[0].size()+cdp->firing_data.size()+cdp->not_firing_data.size());a++)
        {
            temp_id.slack=true;
            temp_id.basic=false;
            temp_id.z=false;
            temp_id.theta=false;
            temp_id.rhs=false;
            temp_id.id=slack_id;
            st.c_id.push_back(temp_id);
            slack_id++;
        }

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=false;
        temp_id.z=true;
        temp_id.theta=false;
        temp_id.id=slack_id;
        st.c_id.push_back(temp_id);
        slack_id++;

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=true;
        temp_id.z=false;
        temp_id.theta=false;
        temp_id.id=slack_id;
        st.c_id.push_back(temp_id);

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=false;
        temp_id.z=false;
        temp_id.theta=true;
        temp_id.id=slack_id;
        st.c_id.push_back(temp_id);

        st.r_id.clear();
        for(int a=0;a<st.c_id.size();a++)
        {
            if(st.c_id[a].slack==true)
            {
                st.r_id.push_back(st.c_id[a]);
            }
        }

        st.slack_var.clear();
        st.basic_var.clear();
        vector<float> temp;
        int x;
        for(int a=0;a<cdp->firing_data.size();a++)
        {
            temp.clear();
            //entering basic variable data
            for(int b=0;b<cdp->firing_data[a].size();b++)
            {   temp.push_back(cdp->firing_data[a][b]);
                temp.push_back(cdp->firing_data[a][b]-2*cdp->firing_data[a][b]);
            }
            st.basic_var.push_back(temp);
            temp.clear();
            //entering slack var data
            for(int b=0;b<st.r_id.size();b++)
            {
                if(b==a)
                {   temp.push_back(-1);x=b;}
                else
                {   temp.push_back(0);}
            }
            //temp.push_back(0);//for z
            //temp.push_back(cdp->lower_firing_constrain_rhs);
            st.slack_var.push_back(temp);

            st.z_col.push_back(0);
            st.rhs.push_back(cdp->lower_firing_constrain_rhs); //modification needs to be done here
        }

        for(int a=0;a<cdp->not_firing_data.size();a++)
        {
            temp.clear();
            //entering basic variable data
            for(int b=0;b<cdp->not_firing_data[a].size();b++)
            {   temp.push_back(cdp->not_firing_data[a][b]);
                temp.push_back(cdp->not_firing_data[a][b]-2*cdp->not_firing_data[a][b]);
            }
            st.basic_var.push_back(temp);
            temp.clear();
            //entering slack var data
            for(int b=0;b<st.r_id.size();b++)
            {
                if((b)==(a+x+1))
                {   temp.push_back(1);}
                else
                {   temp.push_back(0);}
            }
            //temp.push_back(0);//for z
            //temp.push_back(cdp->upper_not_firing_constrain_rhs);
            st.slack_var.push_back(temp);

            st.z_col.push_back(0);
            st.rhs.push_back(cdp->upper_not_firing_constrain_rhs); //modification needs to be done here
        }

        st.z_row.clear();
        //entering z row data
        for(int a=0;a<cdp->objective_function_coefficients.size();a++)
        {
            st.z_row.push_back(cdp->objective_function_coefficients[a]);
            st.z_row.push_back(cdp->objective_function_coefficients[a]-2*cdp->objective_function_coefficients[a]);
        }
        for(int a=0;a<st.r_id.size();a++)
        {   st.z_row.push_back(0);}
        st.z_row.push_back(1);// co_orodianted (z,z)
        st.z_row.push_back(0);//(z,rhs)
        //display_st();
        //cout<<"data size1= "<<cdp->firing_data.size()<<endl;
        //needs modification
        feasible_solution_calculator.start(&st);
        //cout<<"data size2= "<<cdp->firing_data.size()<<endl;
        conflicting_data_id1 conflict_id;
        conflict_id=feasible_solution_calculator.return_conflict_id_pack();

        if(conflict_id.conflict_id_present==true)
        {
            conflicting_data.firing_data.clear();
            conflicting_data.not_firing_data.clear();
            conflicting_data.objective_function_coefficients.clear();
            conflicting_data.weight_matrix.clear();
            if(display_iterations==true)
            {
                cout<<"\nconflicting databefore erasing:\n";
                cout<<"firing data size= "<<cdp->firing_data.size()<<endl;
            }
            if(conflict_id.id.size()==cdp->firing_data.size())//for handling 0:0 bug
            {   
                int firing_data_limit=cdp->firing_data.size()/2;
                int not_firing_data_limit=cdp->not_firing_data.size()/2;
                converted_data_pack cdp_temp;
                for(int a=cdp->firing_data.size()-1;a>=firing_data_limit;a--)//saving the 2nd half firing_data in the conflicting_data
                {
                    conflicting_data.firing_data.push_back(cdp->firing_data[a]);
                    cdp->firing_data.erase(cdp->firing_data.begin()+a);
                }
                for(int a=cdp->not_firing_data.size()-1;a>=not_firing_data_limit;a--)
                {
                    conflicting_data.not_firing_data.push_back(cdp->not_firing_data[a]);
                    cdp->not_firing_data.erase(cdp->not_firing_data.begin()+a);
                }
                conflicting_data.firing_label=cdp->firing_label;
                conflicting_data.firing_neuron_index=cdp->firing_neuron_index;
                start_solver(cdp);
            }
            else
            {
                int count1=0;
                for(int a=conflict_id.id.size()-1;a>=0;a--)
                {
                    //save the conflicting data in a obj of converted data pack.
                    conflicting_data.firing_data.push_back(cdp->firing_data[conflict_id.id[a]]); //copying the conflicting data
                    if(display_iterations==true)
                    {   cout<<"erasing id= "<<conflict_id.id[a]<<",";}
                    cdp->firing_data.erase(cdp->firing_data.begin()+conflict_id.id[a]); //erasing the conflicting data
                    if(display_iterations==true)
                    {   cout<<"firing data size= "<<cdp->firing_data.size()<<endl;}
                    count1++;
                }
                //ratio maintainer
                int current_not_firing_data_size=cdp->not_firing_data.size();
                int current_firing_data_size=cdp->firing_data.size();//                                   .get_ratio(cdp->firing_neuron_index)                        .get_ratio(cdp->firing_neuron_index) //get_ratio function is not required in this version.
                float no_of_notfiring_data_to_be_removed_float=(current_not_firing_data_size*ratio_locker1.get_default_ratio()-current_firing_data_size)/ratio_locker1.get_default_ratio();
                int no_of_notfiring_data_to_be_removed=no_of_notfiring_data_to_be_removed_float;
                    //not firing data are veing removed from cdp and added to conflicting_data
                if(display_iterations==true)
                {
                    cout<<"\nno of not_firing_data to be removed= "<<no_of_notfiring_data_to_be_removed<<endl;
                    cout<<"current not_firing_data size= "<<cdp->not_firing_data.size()<<endl;
                    cout<<"firing_data_size= "<<cdp->firing_data.size()<<endl;
                    cout<<"no of firing data removed= "<<count1<<endl;
                    cout<<"default_ratio= "<<ratio_locker1.get_default_ratio()<<endl;
                    cout<<"cdp->firing_neuron_index= "<<cdp->firing_neuron_index<<endl;
                }
                for(int a=current_not_firing_data_size-1;a>=(current_not_firing_data_size-no_of_notfiring_data_to_be_removed);a--)
                {
                    conflicting_data.not_firing_data.push_back(cdp->not_firing_data[a]);
                    cdp->not_firing_data.erase(cdp->not_firing_data.begin()+a);
                }
                conflicting_data.firing_neuron_index=cdp->firing_neuron_index; //this line is important 
                conflicting_data.firing_label=cdp->firing_label; //this line is important
                if(display_iterations==true)
                {
                    cout<<"\nafter erasing\n";
                    cout<<"firing_size= "<<cdp->firing_data.size()<<" not_firing_size= "<<cdp->not_firing_data.size()<<endl;
                }
                start_solver(cdp); //calling start_solver function with non conflicting data.
            }
            
        }
        else
        {
            cdp->weight_matrix.clear();
            vector<double> weight_matrix_raw;
            weight_matrix_raw.clear();
            //push the raw solution in the converted data pack.
            for(int a=0;a<cdp->objective_function_coefficients.size()*2;a++)
            {
                bool found=false;
                for(int b=0;b<st.r_id.size();b++)
                {
                    if(st.r_id[b].id==a && st.r_id[b].basic==true)
                    {
                        //cdp->weight_matrix.push_back(st.rhs[b]*st.basic_var[b][a]);
                        weight_matrix_raw.push_back(st.rhs[b]*st.basic_var[b][a]);
                        found=true;
                        break;
                    }
                }
                if(found==false)
                {
                    //cdp->weight_matrix.push_back(0);
                    weight_matrix_raw.push_back(0);
                }
            }
            //pushing the calculated solution in the weight matrix
            for(int a=0;a<cdp->objective_function_coefficients.size()*2;a+=2)
            {
                cdp->weight_matrix.push_back(weight_matrix_raw[a]-weight_matrix_raw[a+1]);
            }
            //as cdp is an address of the original cdp obj is already present in the previous function.
        }
    }
    converted_data_pack conflicting_data;
};

class simplex_solver_data_preparation_class
{
    private:
    struct conflicting_data_buffer_vector_struct
    {
        vector<converted_data_pack> conflicting_data_buffer_vector;
    }conflicting_data_buffer_outer;

    ann* network;
    datapack_structure_defination* data_structure;
    converted_data_pack conflicting_data_buffer;
    modified_simplex_solver lpp_solver1;
    vector<converted_data_pack> cdp;
    void cdp_viewer(converted_data_pack* cdp)
    {
        cout<<"firing neuron index= "<<cdp->firing_neuron_index<<"  label= "<<cdp->firing_label<<endl;
        cout<<"objective function coefficient= "<<endl;
        for(int a=0;a<cdp->objective_function_coefficients.size();a++)
        {   cout<<cdp->objective_function_coefficients[a]<<",";}
        cout<<"\n\nfiring data:-"<<endl<<endl;
        for(int a=0;a<cdp->firing_data.size()-100;a++)
        {
            for(int b=0;b<cdp->firing_data[a].size();b++)
            {   cout<<cdp->firing_data[a][b]<<",";}
            cout<<endl;
        }
        cout<<"\n\nnot_firing data:-\n\n";
        for(int a=0;a<cdp->not_firing_data.size()-100;a++)
        {
            for(int b=0;b<cdp->not_firing_data[a].size();b++)
            {   cout<<cdp->not_firing_data[a][b]<<",";}
            cout<<endl;
        }
    }

    void cdp_saver_in_mathematical_format(converted_data_pack* cdp)
    {
        fstream file1("question.txt",ios::out);

        for(int a=0;a<cdp->objective_function_coefficients.size();a++)
        {
            file1<<cdp->objective_function_coefficients[a]<<"c"<<a+1<<"+";
            cout<<cdp->objective_function_coefficients[a]<<",";
        }
        file1<<"\n\n\n";
        for(int a=0;a<cdp->firing_data.size();a++)
        {
            file1<<"\n";
            for(int b=0;b<cdp->firing_data[a].size();b++)
            {
                file1<<cdp->firing_data[a][b]<<"c"<<b+1;
                if(b==cdp->firing_data[a].size()-1)
                {   continue;}
                file1<<"+";
            }
            file1<<">40";
        }
        file1<<"\n\n";
        file1.close();
        ofstream file2("question2.txt",ios::out);
        file2<<"\n\n";
        cout<<" size= "<<cdp->not_firing_data.size();
        for(int a=0;a<cdp->not_firing_data.size();a++)
        {
            file2<<"\n";
            for(int b=0;b<cdp->not_firing_data[a].size();b++)
            {
                file2<<cdp->not_firing_data[a][b]<<"c"<<b+1;
                //cout<<cdp->not_firing_data[a][b]<<"c"<<b+1<<endl;
                if(b==cdp->not_firing_data[a].size()-1)
                {   continue;}
                file2<<"+";
            }
            //cout<<a<<" ";
            file2<<"<10";
        }
    }
    /*void cdp_saver(converted_data_pack* cdp)
    {
        ofstream file2("question2.txt",ios::out);
        file2<<"\n\n";
        cout<<" size= "<<cdp->not_firing_data.size();
        for(int a=0;a<cdp->not_firing_data.size();a++)
        {
            file2<<"\n";
            for(int b=0;b<cdp->not_firing_data[a].size();b++)
            {
                file2<<cdp->not_firing_data[a][b]<<"c"<<b+1;
                //cout<<cdp->not_firing_data[a][b]<<"c"<<b+1<<endl;
                if(b==cdp->not_firing_data[a].size()-1)
                {   continue;}
                file2<<"+";
            }
            //cout<<a<<" ";
            file2<<"<10";
        }
        file2<<"testing hello";
    }*/
    public:
    void lp_solver()
    {
        //the main conflicting_data_outer cleaner
        conflicting_data_buffer_outer.conflicting_data_buffer_vector.clear();
        //ratio locker not needed in this version
        /*if(ratio_locker1.is_locked()==false)
        {
            vector<int> label_data_sizes;
            for(int a=0;a<cdp.size();a++)
            {
                label_data_sizes.push_back(cdp[a].firing_data.size());
            }
            ratio_locker1.set_ratios(label_data_sizes);
        }*/
        /*for(int a=0;a<cdp.size();a++)
        {
            if(cdp[a].firing_data.size()!=cdp[a].not_firing_data.size())
            {
                cout<<"\n\na= "<<a;
                cout<<"\nfds= "<<cdp[a].firing_data.size();
                cout<<"\nnfds= "<<cdp[a].not_firing_data.size();
                cout<<"\nfiring label= "<<cdp[a].firing_label;
            }
        }
        cout<<"\n\ntesting finished!!!!";*/
        //int df;cin>>fg;
        for(int a=0;a<cdp.size();a++)
        {
            shared_block_data_obj.no_of_c_datapacks_completed++;
            //cout<<"\nno_fo_c_datapacks_completed= "<<shared_block_data_obj.no_of_c_datapacks_completed;
            lpp_solver1.start_solver(&cdp[a]);
            if(display_iterations==true)
            {
                cout<<"\nfiring size= "<<cdp[a].firing_data.size()<<"\nnot firing size= "<<cdp[a].not_firing_data.size()<<"\n";
                cout<<"weight matrix: ";
                for(int b=0;b<cdp[a].weight_matrix.size();b++)
                {
                    cout<<cdp[a].weight_matrix[b]<<",";
                }
                cout<<endl;
            }
            //adding the conflicting data to the conflicting data buffer
            //getting the conflicting data from the lpp_solver1 and adding them to the conflicting_data_buffer.
            conflicting_data_buffer.firing_data.insert(conflicting_data_buffer.firing_data.end(),lpp_solver1.conflicting_data.firing_data.begin(),lpp_solver1.conflicting_data.firing_data.end());
            conflicting_data_buffer.not_firing_data.insert(conflicting_data_buffer.not_firing_data.end(),lpp_solver1.conflicting_data.not_firing_data.begin(),lpp_solver1.conflicting_data.not_firing_data.end());
            conflicting_data_buffer.firing_neuron_index=lpp_solver1.conflicting_data.firing_neuron_index; //this line is important //the data here is directily brought from the cdp
            conflicting_data_buffer.firing_label=lpp_solver1.conflicting_data.firing_label; //this line is important //the data here is directily brought from the cdp
            //claner of conflicting_data_buffer_outer.conflicting_data_buffer_vector
            //erasing the conflicting data from the lpp_solver1
            lpp_solver1.conflicting_data.firing_data.clear();
            lpp_solver1.conflicting_data.not_firing_data.clear();
            lpp_solver1.conflicting_data.objective_function_coefficients.clear();
            lpp_solver1.conflicting_data.weight_matrix.clear();

            //lp_optimizer may be added here.
            //handling each of the weight matrix is required.

            // 1. create new path
            //   1a. provide the output neuron id. (training for which output neuron or which label. First neuron = label 2 second neuron = label 4)
            //   2b. push the weight matrix
            network->create_new_path(cdp[a].weight_matrix,cdp[a].firing_neuron_index);
            //enter the label of the conflicting data //I think this process is already complete...... 
            
            //pushing the conflicting_data_buffer of each label to the conflicting_data_buffer_vector.
            if(conflicting_data_buffer.firing_data.size()>0 && conflicting_data_buffer.not_firing_data.size()>0)
            {
                conflicting_data_buffer_outer.conflicting_data_buffer_vector.push_back(conflicting_data_buffer);
            }
            
            //cleanup
            conflicting_data_buffer.weight_matrix.clear();
            conflicting_data_buffer.objective_function_coefficients.clear();
            conflicting_data_buffer.firing_data.clear();
            conflicting_data_buffer.not_firing_data.clear();
        }
        // 2. conflicting datas must be handled here.

        if(conflicting_data_buffer_outer.conflicting_data_buffer_vector.size()>0)
        {
            //objective function coefficient calculation
            int d=0;
            for(int a=0;a<conflicting_data_buffer_outer.conflicting_data_buffer_vector.size();a++)
            {
                conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients.resize(data_structure->no_of_elements_in_each_record);
                for(int b=0;b<data_structure->no_of_elements_in_each_record;b++)
                {
                    for(d=0;d<conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data.size();d++){
                        conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]+=conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data[d][b];
                    }
                    conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]=conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]/d;
                }
            }
            for(int a=0;a<conflicting_data_buffer_outer.conflicting_data_buffer_vector.size();a++)
            {
                if(conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data.size()==0)
                {
                    conflicting_data_buffer_outer.conflicting_data_buffer_vector.erase(conflicting_data_buffer_outer.conflicting_data_buffer_vector.begin()+a);
                }
            }
            cout<<"\nsolving for the conflicting data.....\n";
            cdp=conflicting_data_buffer_outer.conflicting_data_buffer_vector;
            lp_solver();
        }
        else
        {   cout<<"\nfinished training the entire package.\n";}
    }

    simplex_solver_data_preparation_class(vector<converted_data_pack> cdps,datapack_structure_defination* ds,ann* network1)
    {
        network=network1;
        data_structure=ds;
        cdp=cdps;
    }
};
//int training_step_counter=0;

struct network_structure_defination{
    int no_of_input_neuron=0;
    int no_of_output_neuron=0;
    int mean_buffer_size=0;
    vector<float> elements;
};

class core_class{
    private:
    int core_no=0,core_aim=0;
    int required_no_of_threads;
    int predict_progress_bar_numerator=0;//for the predict progress bar
    int predict_progress_bar_denominator=0;//for the predict progress bar
    float data_division=1.5;
    ann network1;
    //simplex_solver_data_preparation_class lpp_solver;
    datapack_structure_defination ds;
    network_structure_defination ns;

    bool check_if_datapack_has_valid_labels(data_package_class* data_pack)//if a label is 0 than it is invalid
    {
        if(data_pack->data.size()>data_pack->labels.size())
        {   return false;}
        else
        {   return true;}
    }

    void save_network()
    {
        time_t theTime = time(NULL);
        struct tm *aTime = localtime(&theTime);
        int day = aTime->tm_mday;
        string day_str= to_string(day);
        if(day<10)
        {   day_str="0"+day_str;}
        int month = aTime->tm_mon + 1; // Month is 0 - 11, add 1 to get a jan-dec 1-12 concept
        string month_str=to_string(month);
        if(month<10)
        {   month_str="0"+month_str;}
        int year = aTime->tm_year + 1900; // Year is # years since 1900
        string year_str=to_string(year);
        time_t now = time(0);
        // Convert now to tm struct for local timezone
        tm* localtm = localtime(&now);
        int hr,min,sec;
        hr=localtm->tm_hour;
        min=localtm->tm_min;
        sec=localtm->tm_sec;
        string hr_str=to_string(hr),min_str=to_string(min),sec_str=to_string(sec);
        if(hr<10)
        {   hr_str="0"+hr_str;}
        if(min<10)
        {   min_str="0"+min_str;}
        if(sec<10)
        {   sec_str="0"+sec_str;}
        string net_savefile_id="";//core_aim,core_no,day,month,year
        string core_aim_str=to_string(core_aim);
        string core_no_str=to_string(core_no);
        net_savefile_id=core_aim_str+core_no_str+year_str+month_str+day_str+hr_str+min_str+sec_str;
        string net_savefile_name="network-"+net_savefile_id+".csv";
        ofstream file1(net_savefile_name,ios::out);
        file1<<"FILE_NAME:,\n";
        file1<<"name=,"<<net_savefile_name<<",\n";
        file1<<"BASIC_SAVEFILE_INFO:,\n";
        file1<<",network_save_id,core_aim,core_no,year,month,day,hour,minute,sec,\n";
        file1<<"network_save_id,"<<net_savefile_id<<","<<core_aim<<","<<core_no<<","<<year_str<<","<<month_str<<","<<day_str<<","<<hr_str<<","<<min_str<<","<<sec_str<<",\n";
        file1<<"BASIC_NETWORK_INFO:,";
        file1<<"net_info_category,no_of_input_neuron,no_of_output_neuron,\n";
        file1<<",net_info_category,"<<network1.input_neuron_size()<<","<<network1.output_neuron_size()<<",\n";
        file1<<"DATA_LABEL_AND_OUTPUT_NEURON_INDEX:,\n";
        file1<<"[data_label~output_neuron_index],";
        for(int a=0;a<ds.elements.size();a++)
        {
            file1<<ds.elements[a]<<"~"<<a<<",";
        }
        file1<<"\n";
        file1<<"PATH_INFO:,\n";
        file1<<"no_of_path=,"<<network1.return_path_size()<<",\n";
        file1<<"path_id,";
        for(int a=0;a<network1.path[0].weight_matrix.size();a++)
        {
            int weight_index=a;
            string weight_index_str=to_string(weight_index);
            weight_index_str="w"+weight_index_str;
            file1<<weight_index_str<<",";
        }
        file1<<"input_neuron_id,output_neuron_id\n";
        for(int a=0;a<network1.path.size();a++)
        {
            file1<<network1.path[a].path_id<<",";
            for(int b=0;b<network1.path[a].weight_matrix.size();b++)
            {
                file1<<network1.path[a].weight_matrix[b]<<",";
            }
            file1<<"[|";
            for(int b=0;b<network1.path[a].input_neuron_id.size();b++)
            {
                file1<<network1.path[a].input_neuron_id[b]<<"|";
            }
            file1<<"],";
            file1<<network1.path[a].output_neuron_id<<",\n";
        }
        file1.close();
    }
    
    bool strcasestr(string str,string substr)
    {
        transform(str.begin(), str.end(), str.begin(),ptr_fun<int, int>(toupper));
        transform(substr.begin(), substr.end(), substr.begin(),ptr_fun<int, int>(toupper));
        if(str.find(substr) != string::npos)
        {   return true;}
        else 
        {   return false;}
    }

    string workingdir() //checking required
    {
        char buf[256];
        GetCurrentDirectoryA(256, buf);
        return std::string(buf) + '\\';
    }

    bool load_network_if_available(int core_aim=0,int core_no=0,bool file_name_received=false,string file_name="")
    {
        
        string file_name_local;
        bool file_found=false;
        int weight_count=0;//weight_count
        int no_of_input_neuron,no_of_output_neuron;
        vector<float> elements;
        if(file_name_received==false)
        {        
            string tempstr=workingdir();
            int dir_size=tempstr.size()*2;
            char ch[2]={'\0'},dir[dir_size];
            for(int a=0;a<strlen(dir);a++)
            {   dir[a]={'\0'};}
            for(int a=0;a<tempstr.length();a++)
            {
                if(tempstr.at(a)=='\\')
                {   tempstr[a]='/';}
                ch[0]=tempstr.at(a);
                ch[1]='\0';
                strcat(dir,ch);
            }
            cout<<dir;

            char filename_sub_str[8]="network";
            vector<string> network_save_file_list;
            network_save_file_list.clear();

            QDir directory(dir);
            QStringList files = directory.entryList(QStringList() << "*",QDir::Files);
            foreach(QString filename, files)
            {
                string str=filename.toStdString();
                if(strcasestr(str,filename_sub_str)==true)
                {   network_save_file_list.push_back(str);}
            }

            vector<ifstream> input_file_streams(network_save_file_list.size());
            for(int a=0;a<network_save_file_list.size();a++)
            {   input_file_streams[a].open(network_save_file_list[a],ios::in);}
            int input_file_stream_index;
            for(int a=0;a<input_file_streams.size();a++)
            {
                int core_aim_read=7,core_no_read=5;
                while(input_file_streams[a])
                {
                    string line;
                    input_file_streams[a]>>line;
                    if(input_file_streams[a].eof())
                    {   break;}
                    char line_arr[line.length()];
                    bool core_no_read_collected=false,core_aim_read_collected=false;
                    strcpy(line_arr,line.c_str());
                    if(strcasestr(line_arr,"BASIC_SAVEFILE_INFO:"))
                    {
                        input_file_streams[a]>>line;
                        int core_aim_index,core_no_index,word_index=0;
                        bool core_no_found=false,core_aim_found=false;
                        char ch[2]={'\0'},ch_arr[20]={'\0'};
                        for(int b=0;b<line.length();b++)
                        {
                            if(line.at(b)==',')
                            {
                                //float val = atof(num_char);  
                                word_index++;
                                if(strcasestr(ch_arr,"core_no"))
                                {
                                    //cout<<"core_no= "<<word_index<<endl;
                                    core_no_index=word_index;
                                    core_no_found=true;
                                }
                                if(strcasestr(ch_arr,"core_aim"))
                                {
                                    //cout<<"core_aim= "<<word_index;
                                    core_aim_index=word_index;
                                    core_aim_found=true;
                                }
                                if(core_aim_found==true && core_no_found==true)
                                {   break;}
                                for(int b=0;b<20;b++){
                                    ch_arr[b]='\0';
                                }
                                continue;
                            }
                            ch[0]=line.at(b);
                            ch[1]='\0';
                            strcat(ch_arr,ch);
                        }
                        if(core_aim_found==true && core_no_found==true)
                        {
                            input_file_streams[a]>>line;
                            word_index=0;
                            for(int b=0;b<line.length();b++)
                            {
                                if(line.at(b)==',')
                                {
                                    //float val = atof(num_char);  
                                    word_index++;
                                    if(word_index==core_aim_index)
                                    {   
                                        core_aim_read=atoi(ch_arr);
                                        //cout<<"core_aim= "<<core_aim_read<<endl;
                                    }
                                    if(word_index==core_no_index)
                                    {   
                                        core_no_read=atoi(ch_arr);
                                        //cout<<"core_no= "<<core_no_read<<endl;
                                    }
                                    for(int b=0;b<20;b++){
                                        ch_arr[b]='\0';
                                    }
                                    continue;
                                }
                                ch[0]=line.at(b);
                                ch[1]='\0';
                                strcat(ch_arr,ch);
                            }

                        }
                    }
                    if(core_no_read_collected==true,core_aim_read_collected==true)
                    {   break;}
                }
                //cout<<"core_aim_read="<<core_aim_read<<" core_aim="<<core_aim<<endl;
                //cout<<"core_no_read="<<core_no_read<<" core_no="<<core_aim<<endl;
                if(core_aim==core_aim_read && core_no==core_no_read)
                {   
                    input_file_stream_index=a;
                    file_found=true;
                    break;
                }
            }
            for(int a=0;a<input_file_streams.size();a++)
            {   input_file_streams[a].close();}
            if(file_found==true)
            {   file_name_local=network_save_file_list[input_file_stream_index];}
        }
        if(file_found==true || file_name_received==true)
        {
            if(file_name_received==true)
            {   file_name_local=file_name;}
            ifstream input_net_savefile(file_name_local,ios::in);
            bool path_info_found=false;int p=0;
            while(input_net_savefile)
            {
                string line;
                input_net_savefile>>line;
                if(input_net_savefile.eof())
                {   break;}
                char line_arr[line.length()];
                strcpy(line_arr,line.c_str());
                int no_of_paths;
                if(path_info_found==true)
                {
                    p++;//reading the paths data.
                    char arr_temp[line.length()],ch[2];
                    int output_neuron_id;
                    vector<float> weight_matrix;
                    weight_matrix.clear();
                    int index=0;
                    for(int a=0;a<line.length();a++)
                    {
                        if(line.at(a)==',')
                        {
                            index++;
                            if(index>1 && (index-1)<=weight_count )
                            {
                                weight_matrix.push_back(atof(arr_temp));
                            }
                            if(index==(3+weight_count))
                            {
                                output_neuron_id=atof(arr_temp);
                            }
                            for(int b=0;b<line.length();b++){
                                arr_temp[b]='\0';
                            }
                            continue;
                        }
                        ch[0]=line.at(a);
                        ch[1]='\0';
                        strcat(arr_temp,ch);
                    }
                    //point for entering the data in the network
                    /*for(int a=0;a<weight_matrix.size();a++)
                    {
                        cout<<weight_matrix[a]<<",";
                    }
                    cout<<" output_neuron= "<<output_neuron_id;
                    cout<<endl;*/
                    network1.create_new_path(weight_matrix,output_neuron_id);
                }
                if(strcasestr(line_arr,"BASIC_NETWORK_INFO"))//needed to be added in the core_class function
                {
                    input_net_savefile>>line;
                    char arr_temp[line.length()],ch[2];
                    int word_index=0;
                    for(int a=0;a<line.length();a++)
                    {
                        if(line.at(a)==',')
                        {
                            word_index++;
                            if(word_index==3)
                            {
                                no_of_input_neuron=atoi(arr_temp);//cout<<"arr_temp= "<<arr_temp;
                            }
                            else if(word_index==4)
                            {
                                no_of_output_neuron=atoi(arr_temp);
                            }
                            for(int b=0;b<line.length();b++)
                            {   arr_temp[b]='\0';}
                            continue;
                        }
                        ch[0]=line.at(a);
                        ch[1]='\0';
                        strcat(arr_temp,ch);
                    }
                }
                if(strcasestr(line_arr,"DATA_LABEL_AND_OUTPUT_NEURON_INDEX:"))
                {
                    input_net_savefile>>line;
                    char arr_temp[line.length()],ch[2];
                    int word_index=0;
                    for(int a=0;a<line.length();a++)
                    {
                        if(line.at(a)==',')
                        {
                            word_index++;
                            if(word_index>1)
                            {
                                bool arr_temp_cleaner=false;
                                for(int b=0;b<strlen(arr_temp);b++)
                                {
                                    if(arr_temp[b]=='~')
                                    {   arr_temp_cleaner=true;}
                                    if(arr_temp_cleaner==true)
                                    {   arr_temp[b]='\0';}
                                }
                                elements.push_back(atof(arr_temp));
                            }
                            for(int b=0;b<line.length();b++)
                            {   arr_temp[b]='\0';}
                            continue;
                        }
                        ch[0]=line.at(a);
                        ch[1]='\0';
                        strcat(arr_temp,ch);
                    }
                    network1.set_elements_vector(elements);
                }
                if(strcasestr(line_arr,"PATH_INFO:"))
                {
                    input_net_savefile>>line;
                    char arr_temp[line.length()],ch[2];
                    int word_index=0;
                    for(int a=0;a<line.length();a++)
                    {
                        if(line.at(a)==',')
                        {
                            word_index++;
                            if(word_index==2)
                            {   
                                no_of_paths=atoi(arr_temp);
                            }
                            for(int b=0;b<line.length();b++){
                                arr_temp[b]='\0';
                            }
                            continue;
                        }
                        ch[0]=line.at(a);
                        ch[1]='\0';
                        strcat(arr_temp,ch);
                    }
                    //cout<<"no_of_paths= "<<no_of_paths;
                    input_net_savefile>>line;
                    char arr_temp1[line.length()],ch1[2];
                    for(int b=0;b<line.length();b++)
                    {
                        arr_temp1[b]='\0';
                    }
                    for(int a=0;a<line.length();a++)
                    {
                        if(line.at(a)==',')
                        {
                            //cout<<"arr_temp1= "<<arr_temp1<<endl;
                            if(strcasestr(arr_temp1,"w"))
                            {
                                weight_count++;
                            }
                            for(int b=0;b<line.length();b++){
                                arr_temp1[b]='\0';
                            }
                            continue;
                        }
                        ch1[0]=line.at(a);
                        ch1[1]='\0';
                        strcat(arr_temp1,ch1);
                    }
                    path_info_found=true;
                    //cout<<"weight_count= "<<weight_count;
                    //int g;cin>>g;
                }
            }
            //the below lines does the work of network_structure_modifier() function
            network1.set_no_of_input_neuron(no_of_input_neuron);
            network1.set_no_of_output_neuron(no_of_output_neuron);
            network_analyzer();//initialization of ns
            //cout<<"p="<<p;
            return true;
        }
        else
        {   return false;}
    }

    void network_analyzer()
    {
        ns.no_of_input_neuron=network1.input_neuron_size();
        ns.no_of_output_neuron=network1.output_neuron_size();
        ns.elements=network1.elements;
        ns.mean_buffer_size=network1.input_neurons.size();//what the fuck is this????
    }

    void datapack_analyzer(data_package_class* data_pack)//checked, it fills up the datapack_structure_defination ds. 
    {
        if(data_pack->analyze_status==false)
        {
            ds.no_of_elements_in_each_record=data_pack->data[0].size();
            vector<int> labels;
            bool found=false;
            for(int a=0;a<data_pack->data.size();a++)
            {
                found=false;
                for(int b=0;b<labels.size();b++)
                {
                    if(labels[b]==data_pack->labels[a])
                    {
                        found=true;
                        break;
                    }
                }
                if(found==false)
                {
                    labels.push_back(data_pack->labels[a]);
                }
            }
            ds.no_of_labels=labels.size();
            for(int a=0;a<labels.size();a++)
            {
                ds.elements.push_back(labels[a]);
            }
            data_pack->analyze_status=true;
        }
    }

    void network_structure_modifier()//this function may need further improvements
    {
        try{
            if(ns.no_of_input_neuron>ds.no_of_elements_in_each_record)
            {
                throw("network has more neuron than required by the data");//needs working here. UNDER CONSTRUCTION.
                save_network();
            }
            else
            {
                network1.set_no_of_input_neuron(ds.no_of_elements_in_each_record);
                network1.set_no_of_output_neuron(ds.no_of_labels);
            }
        }
        catch(string s)
        {   cout<<s<<endl;}
    }

    struct shuffling_data{
    vector<float> temp_data;
    int temp_label;
    };

    void shuffler(filtered_data* f_data)
    {
        vector<shuffling_data> shuffling_data_temp_vector;
        shuffling_data shuffling_data_temp;
        shuffling_data_temp_vector.clear();
        //pushing the data the the shuffling vector
        for(int a=0;a<f_data->data.size();a++)
        {
            shuffling_data_temp.temp_data=f_data->data[a];
            shuffling_data_temp.temp_label=f_data->label;
            shuffling_data_temp_vector.push_back(shuffling_data_temp);
        }
        //shuffeling the data
        random_shuffle(shuffling_data_temp_vector.begin(),shuffling_data_temp_vector.end());
        //pushing the data in the f_data
        f_data->data.clear();
        for(int a=0;a<shuffling_data_temp_vector.size();a++)
        {
            f_data->data.push_back(shuffling_data_temp_vector[a].temp_data);
            f_data->label=shuffling_data_temp_vector[a].temp_label;
        }
    }

    void f_data_viewer(string str,vector<filtered_data> f_data)
    {
        cout<<"\n"<<str<<endl;
        for(int a=0;a<f_data.size();a++)
        {
            cout<<"\nlabel= "<<f_data[a].label<<" data_size= "<<f_data[a].data.size();
        }
    }

    void filter(data_package_class* data_pack,int train_test_predict)
    {
        filtered_data f_data;
        vector<filtered_data> f_data_vector;
        for(int a=0;a<ds.elements.size();a++)
        {
            f_data.data.clear();
            for(int b=0;b<data_pack->labels.size();b++)
            {
                if(ds.elements[a]==data_pack->labels[b])
                {
                    f_data.data.push_back(data_pack->data[b]);
                    f_data.label=data_pack->labels[b];
                }
            }
            f_data_vector.push_back(f_data);
        }
        //calling the shuffler
        for(int a=0;a<f_data_vector.size();a++)
        {
            shuffler(&f_data_vector[a]);//has same size as that is of no of element
        }
        //copying the data to f_train_data and f_test_data
        for(int a=0;a<f_data_vector.size();a++)
        {
            filtered_data f_data_temp;
            //copying data to f_train_data
            int limit=0;
            if(train_test_predict==0)//for training
            {   limit=f_data_vector[a].data.size();}
            else if(train_test_predict==1)//for testing
            {   limit=f_data_vector[a].data.size()/data_division;}
            //for(int b=0;b<f_data_vector[a].data.size()/data_division;b++)
            for(int b=0;b<limit;b++)
            {
                f_data_temp.data.push_back(f_data_vector[a].data[b]);
            }
            f_data_temp.label=f_data_vector[a].label;
            f_train_data.push_back(f_data_temp);
            f_data_temp.data.clear();
            //copying data to f_test_data
            int begin=0;
            if(train_test_predict==0)//for training 
            {   begin=f_data_vector[a].data.size();}
            else if(train_test_predict==1)//for testing
            {   begin=f_data_vector[a].data.size()/data_division;}
            //for(int b=f_data_vector[a].data.size()/data_division;b<f_data_vector[a].data.size();b++)
            for(int b=begin;b<f_data_vector[a].data.size();b++)
            {
                f_data_temp.data.push_back(f_data_vector[a].data[b]);
            }
            f_data_temp.label=f_data_vector[a].label;
            f_test_data.push_back(f_data_temp);
            f_data_temp.data.clear();
        }
        //f_data_viewer("train_data",f_train_data);
        //f_data_viewer("test_data",f_test_data);
        //copying the data to the data_pack s
        for(int a=0;a<f_data_vector.size();a++)
        {
            //copying data to testing data
            int begin=0;
            if(train_test_predict==0)//for training 
            {   begin=f_data_vector[a].data.size();}
            else if(train_test_predict==1)//for testing
            {   begin=f_data_vector[a].data.size()/data_division;}
            //for(int b=f_data_vector[a].data.size()/data_division;b<f_data_vector[a].data.size();b++)
            for(int b=begin;b<f_data_vector[a].data.size();b++)
            {
                vector<float> temp;
                for(int c=0;c<f_data_vector[a].data[b].size();c++)
                {
                    temp.push_back(f_data_vector[a].data[b][c]);
                }
                test_data.data.push_back(temp);
                temp.clear();
                test_data.labels.push_back(f_data_vector[a].label);
                //f_data_vector.clear(); //keep this line commented out else the sized of each label data gets zero
            }
        }
    }

    void big_c_datapack_handler(vector<converted_data_pack> &cdp)//passing the vector by reference //this function might be a temporary offer //this is for preventing 0:0 bug
    {
        int limit=300;
        converted_data_pack cdp_temp1,cdp_temp2;
        vector<converted_data_pack> cdp_vect_temp;
        for(int a=cdp.size()-1;a>=0;a--)
        {
            if(cdp[a].firing_data.size()>limit)
            {
                cdp_temp1=cdp[a];
                cdp.erase(cdp.begin()+a);
                int begin=0,end=0;
                bool end_reached=false;
                while(end_reached==false)
                {
                    cdp_temp2.firing_data.clear();
                    cdp_temp2.not_firing_data.clear();
                    cdp_temp2.objective_function_coefficients.clear();
                    begin=end;
                    end=begin+limit;
                    if(end>=cdp_temp1.firing_data.size())
                    {   end=cdp_temp1.firing_data.size();end_reached=true;}
                    cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+begin,cdp_temp1.firing_data.begin()+end);
                    if(end_reached==true)
                    {   end=cdp_temp1.not_firing_data.size();}
                    else if(end>=cdp_temp1.not_firing_data.size())
                    {
                        int end2=cdp_temp1.firing_data.size();
                        cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+end,cdp_temp1.firing_data.begin()+end2);
                        end=cdp_temp1.not_firing_data.size();
                        end_reached=true;
                    }
                    cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+begin,cdp_temp1.not_firing_data.begin()+end);
                    cdp_temp2.firing_label=cdp_temp1.firing_label;
                    cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                    cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                    cdp_vect_temp.push_back(cdp_temp2);
                }
                cdp_temp1.firing_data.clear();
                cdp_temp1.not_firing_data.clear();
                cdp_temp1.objective_function_coefficients.clear();
            }
        }
        cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
        string str_for_body="";
        str_for_body.clear();
        str_for_body.append("\n\ncdp size after erasing in big data handler = ");
        str_for_body.append(to_string(cdp.size()));
        str_for_body.append(", cdp_vect_temp size= ");
        str_for_body.append(to_string(cdp_vect_temp.size()));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\n\ncdp size after erasing in big data handler = "<<cdp.size();
        cout<<", cdp_vect_temp size= "<<cdp_vect_temp.size();
        cdp_vect_temp.clear();
        
        for(int a=0;a<cdp.size();a++)
        {
            int difference=cdp[a].firing_data.size()-cdp[a].not_firing_data.size();
            if(abs(difference)>10)
            {
                cdp_temp1.firing_data.clear();
                cdp_temp1.not_firing_data.clear();
                cdp_temp1.objective_function_coefficients.clear();
                cdp_temp2.firing_data.clear();
                cdp_temp2.not_firing_data.clear();
                cdp_temp2.objective_function_coefficients.clear();
                cdp_temp1=cdp[a];
                cdp.erase(cdp.begin()+a);
                while(abs(difference)>10)
                {
                    if(difference<0)
                    {
                        limit=cdp_temp1.firing_data.size();
                        cdp_temp2.firing_data=cdp_temp1.firing_data;
                        cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                        cdp_temp2.firing_label=cdp_temp1.firing_label;
                        cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                        cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                        cdp_temp1.not_firing_data.erase(cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                        cdp_vect_temp.push_back(cdp_temp2);
                    }
                    else if(difference>0)
                    {
                        limit=cdp_temp1.not_firing_data.size();
                        cdp_temp2.not_firing_data=cdp_temp1.not_firing_data;
                        cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                        cdp_temp2.firing_label=cdp_temp1.firing_label;
                        cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                        cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                        cdp_temp1.firing_data.erase(cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                        cdp_vect_temp.push_back(cdp_temp2);
                    }
                    difference=cdp_temp1.firing_data.size()-cdp_temp1.not_firing_data.size();
                    cdp_temp2.firing_data.clear();
                    cdp_temp2.not_firing_data.clear();
                    cdp_temp2.objective_function_coefficients.clear();
                }
                cdp.push_back(cdp_temp1);
            }
        }
        cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
        str_for_body.clear();
        str_for_body.append("\ncdp size after stabilizing extreme ratios = ");
        str_for_body.append(to_string(cdp.size()));
        str_for_body.append(", cdp_vect_temp size= ");
        str_for_body.append(to_string(cdp_vect_temp.size()));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\ncdp size after stabilizing extreme ratios = "<<cdp.size();
        cout<<", cdp_vect_temp size= "<<cdp_vect_temp.size();
    }

    void simplex_solver_data_entry_point(vector<filtered_data> f_data_pack,datapack_structure_defination* ds,ann* network1)//checking required threads
    {
        vector<converted_data_pack> c_datapacks;
        converted_data_pack c_datapack;
        //ratio maintance and packing data in c_datapacks.
        int sum_total_training_data=0;
        for(int a=0;a<f_data_pack.size();a++)
        {   sum_total_training_data=sum_total_training_data+f_data_pack[a].data.size();}
        string str_for_body="";
        str_for_body.clear();
        str_for_body.append("\nsize of training data set= ");
        str_for_body.append(to_string(sum_total_training_data));
        str_for_body.append("\n");
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\nsize of training data set= "<<sum_total_training_data<<endl;
        c_datapacks.clear(); //for asured cleaniness
        for(int a=0;a<f_data_pack.size();a++)
        {
            str_for_body.clear();
            str_for_body.append("packing data for label= ");
            str_for_body.append(to_string(f_data_pack[a].label));
            str_for_body.append("\n");
            body_engine_communication_data_obj.add_message(str_for_body);
            cout<<"packing data for label= "<<f_data_pack[a].label<<endl;
            //determining the c_data_pack critical info
            int sum_total_not_firing_data=sum_total_training_data-f_data_pack[a].data.size();
            int no_of_c_data_packs_needed=0,no_of_not_firing_data_in_each_pack=0,no_of_firing_data_in_each_pack=0;
            int additional_firing_data_in_the_last_datapack=0,additional_not_firing_data_in_the_last_datapack=0;
            if(sum_total_not_firing_data>=f_data_pack[a].data.size())//for not firing data > firing data
            {
                while(sum_total_not_firing_data>=f_data_pack[a].data.size())
                {
                    sum_total_not_firing_data=sum_total_not_firing_data-f_data_pack[a].data.size();//cout<<"check2";
                    no_of_c_data_packs_needed++;
                }
                int rem1=sum_total_not_firing_data;
                no_of_firing_data_in_each_pack=f_data_pack[a].data.size();
                no_of_not_firing_data_in_each_pack=f_data_pack[a].data.size()+rem1/no_of_c_data_packs_needed;
                additional_not_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
            }
            else if(sum_total_not_firing_data<f_data_pack[a].data.size()) //for firing data more than not firing data
            {
                int sum_total_firing_data=f_data_pack[a].data.size();
                while(sum_total_firing_data>=sum_total_not_firing_data)
                {
                    sum_total_firing_data=sum_total_firing_data-sum_total_not_firing_data;//cout<<"check3";
                    no_of_c_data_packs_needed++;
                }
                int rem1=sum_total_firing_data;
                no_of_not_firing_data_in_each_pack=sum_total_not_firing_data;
                no_of_firing_data_in_each_pack=sum_total_not_firing_data+rem1/no_of_c_data_packs_needed;
                additional_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
            }
            
            //packaging the data
            if(no_of_firing_data_in_each_pack==f_data_pack[a].data.size())//this means firing data < not firing data
            {
                int no_of_packages_created=0;
                int initial_value=0,final_value=0;
                vector<vector<float>> not_firing_data_temp;
                not_firing_data_temp.clear();
                //copying all the not firing data in not_firing_data_temp
                for(int b=0;b<f_data_pack.size();b++)
                {
                    if(b!=a)
                    {
                        for(int c=0;c<f_data_pack[b].data.size();c++)
                        {   not_firing_data_temp.push_back(f_data_pack[b].data[c]);}
                    }
                }
                while(no_of_packages_created!=no_of_c_data_packs_needed)
                {
                    //clearing the buffers
                    c_datapack.firing_data.clear();
                    c_datapack.not_firing_data.clear();
                    c_datapack.objective_function_coefficients.clear();
                    c_datapack.weight_matrix.clear();
                    //packing the firing data
                    for(int b=0;b<f_data_pack[a].data.size();b++)
                    {   c_datapack.firing_data.push_back(f_data_pack[a].data[b]);}
                    //packing not firing data
                    initial_value=final_value;
                    final_value=final_value+no_of_not_firing_data_in_each_pack;
                    if(no_of_packages_created==no_of_c_data_packs_needed-1)//for the last package
                    {   final_value=final_value+additional_not_firing_data_in_the_last_datapack;}
                    //cout<<"\nnot_firing_data_temp size= "<<not_firing_data_temp.size();
                    //cout<<"\nfinal_value= "<<final_value<<" initial_value= "<<initial_value;
                    //cout<<"\nadditional_not_firing_data_in_the_last_datapack= "<<additional_not_firing_data_in_the_last_datapack;
                    for(int b=initial_value;b<final_value;b++)
                    {
                        c_datapack.not_firing_data.push_back(not_firing_data_temp[b]);
                    }
                    //setting up the label and output neuron index
                    c_datapack.firing_label=f_data_pack[a].label;
                    c_datapack.firing_neuron_index=a;
                    //setting up the objective function coefficient 
                    for(int b=0;b<c_datapack.firing_data[0].size();b++)
                    {
                        float summation=0;
                        for(int c=0;c<c_datapack.firing_data.size();c++)
                        {   summation=summation+c_datapack.firing_data[c][b];}
                        c_datapack.objective_function_coefficients.push_back(summation);
                    }
                    //pushing the c_datapack in c_datapacks vector
                    c_datapacks.push_back(c_datapack);
                    no_of_packages_created++;
                }
            }
            else if(no_of_not_firing_data_in_each_pack==sum_total_not_firing_data)//this means firing data > not firing data
            {
                int no_of_packages_created=0;
                int initial_value=0,final_value=0;
                while(no_of_packages_created!=no_of_c_data_packs_needed)
                {
                    //clearing the buffers
                    c_datapack.firing_data.clear();
                    c_datapack.not_firing_data.clear();
                    c_datapack.objective_function_coefficients.clear();
                    c_datapack.weight_matrix.clear();
                    //packing the firing data
                    initial_value=final_value;
                    final_value=final_value+no_of_firing_data_in_each_pack;
                    if(no_of_packages_created==no_of_c_data_packs_needed-1)
                    {   final_value=final_value+additional_firing_data_in_the_last_datapack;}
                    for(int b=initial_value;b<final_value;b++)
                    {
                        c_datapack.firing_data.push_back(f_data_pack[a].data[b]);
                    }
                    //packing the not firing data
                    for(int b=0;b<f_data_pack.size();b++)
                    {
                        if(b!=a)
                        {
                            for(int c=0;c<f_data_pack[b].data.size();c++)
                            {   c_datapack.not_firing_data.push_back(f_data_pack[b].data[c]);}
                        }
                    }
                    //setting up the label and output neuron index
                    c_datapack.firing_label=f_data_pack[a].label;
                    c_datapack.firing_neuron_index=a;
                    //setting up the objective function coefficient
                    for(int b=0;b<c_datapack.firing_data[0].size();b++)
                    {
                        float summation=0;
                        for(int c=0;c<c_datapack.firing_data.size();c++)
                        {   summation=summation+c_datapack.firing_data[c][b];}
                        c_datapack.objective_function_coefficients.push_back(summation);
                    }
                    //pushing the c_datapack in c_datapacks vector
                    c_datapacks.push_back(c_datapack);
                    no_of_packages_created++;
                }
            }
        }
        str_for_body.clear();
        str_for_body.append("finished packaging data in c_datapacks.");
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"finished packaging data in c_datapacks.";
        str_for_body.clear();
        str_for_body.append("\ntotal no of c_data_packs= ");
        str_for_body.append(to_string(c_datapacks.size()));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\ntotal no of c_data_packs= "<<c_datapacks.size();
        big_c_datapack_handler(c_datapacks);//for handling c_datapack with huge data which may create full conlflict senarios.
        str_for_body.clear();
        str_for_body.append("\ntotal no of c_data_packs after big c_datapacks handling= ");
        str_for_body.append(to_string(c_datapacks.size()));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\ntotal no of c_data_packs after big c_datapacks handling= "<<c_datapacks.size();
        
        //required_no_of_threads=2;//15
        set_the_no_of_threads_required(c_datapacks.size());
        vector<vector<converted_data_pack>> c_datapacks_vector;
        str_for_body.clear();
        str_for_body.append("\narranging c_datapacks for ");
        str_for_body.append(to_string(required_no_of_threads));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\narranging c_datapacks for "<<required_no_of_threads<<" threads..........";
        c_data_packs_division_for_multi_threading(c_datapacks_vector,c_datapacks,required_no_of_threads);
        //point1:
        if(required_no_of_threads!=c_datapacks_vector.size())
        {
            str_for_body.clear();
            str_for_body.append("\n\nERROR!!! failed to set a valid number of threads. Setting the value to ");
            str_for_body.append(to_string(c_datapacks_vector.size()));                        
            body_engine_communication_data_obj.add_message(str_for_body);

            required_no_of_threads=c_datapacks_vector.size();
            cout<<"\n\nERROR!!! failed to set a valid number of threads. Setting the value to "<<c_datapacks_vector.size();
            //set_the_no_of_threads_required(0,true);//here 0 is just a random value to make this function call work
            //goto point1;
            str_for_body.clear();
            str_for_body.append("\nSetting the no of threads to ");
            str_for_body.append(to_string(required_no_of_threads));
            body_engine_communication_data_obj.add_message(str_for_body);
        }       
        vector<simplex_solver_data_preparation_class> lpp_solver_vec;
        for(int a=0;a<c_datapacks_vector.size();a++)
        {
            simplex_solver_data_preparation_class *lpp_solver=new simplex_solver_data_preparation_class(c_datapacks_vector[a],ds,network1);//initializing the obj of the class   
            lpp_solver_vec.push_back(*lpp_solver);
        }
        //vector<pthread_t> threadIds(required_no_of_threads);//thread declaration for linux
        vector<thread*> thread_vec;//threads for windows
        //pthread_t progress_display_thread;
        //vector<int> error(required_no_of_threads);
        str_for_body.clear();
        str_for_body.append(" lpp_solver_vec size=");
        str_for_body.append(to_string(lpp_solver_vec.size()));
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<" lpp_solver_vec size="<<lpp_solver_vec.size();
        //cin.ignore(1024, '\n');
        //cout << "\n\nPress enter to continue...";
        //cin.get();
        time_t begin=time(0);
        //lpp solvers will start now.........
        for(int a=0;a<required_no_of_threads;a++)
        {
            thread *t=new thread(&simplex_solver_data_preparation_class::lp_solver,&lpp_solver_vec[a]);//thread creator for windows
            thread_vec.push_back(t);
        }

        for(int a=0;a<required_no_of_threads;a++)
        {   thread_vec[a]->join();}

        time_t end=time(0);
        body_engine_communication_data_obj.add_message("\nTASK COMPLETE");
        str_for_body.clear();
        str_for_body.append("\ntime taken for training: ");
        str_for_body.append(to_string(end-begin));
        str_for_body.append(" sec");
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\ntime taken for training: "<<end-begin;
    }

    int size_of_c_datapacks_vector(vector<converted_data_pack> &c_datapacks)
    {
        int sum=0;
        for(int a=0;a<c_datapacks.size();a++)
        {   sum+=c_datapacks[a].firing_data.size();}
        return sum;
    }

    void c_data_packs_division_for_multi_threading(vector<vector<converted_data_pack>> &c_datapacks_vector,vector<converted_data_pack> &c_datapacks,int no_of_threads)
    {
        int total_data=0;
        for(int a=0;a<c_datapacks.size();a++)
        {   total_data+=c_datapacks[a].firing_data.size();}
        int data_for_each_thread=total_data/no_of_threads;
        vector<converted_data_pack> c_datapacks_new;
        c_datapacks_new.clear();
        c_datapacks_vector.clear();
        cout<<"\nno of c_datapacks= "<<c_datapacks.size();
        for(int a=0;a<no_of_threads;a++)
        {
            int b=c_datapacks.size()-1;
            if(b<0)
            {   continue;}
            while(size_of_c_datapacks_vector(c_datapacks_new)<data_for_each_thread)
            {                
                c_datapacks_new.push_back(c_datapacks[b]);
                c_datapacks.erase(c_datapacks.begin()+b);
                b--;
                if(b<0)
                {   break;}
            }
            c_datapacks_vector.push_back(c_datapacks_new);
            c_datapacks_new.clear();
        }
        int sum=0;
        for(int a=0;a<c_datapacks_vector.size();a++)
        {   sum+=c_datapacks_vector[a].size();
            cout<<"\na= "<<a<<" size= "<<c_datapacks_vector[a].size();
        }
        cout<<"\nno of c_datapacks in c_datapacks_vector= "<<sum;
        shared_block_data_obj.total_c_datapacks=sum;
        cout<<"\nc_data_packs size= "<<c_datapacks.size();
    }

    bool ds_ns_have_same_labels()
    {
        if(ds.elements.size()<=ns.elements.size())
        {   
            int count=0;
            for(int a=0;a<ds.elements.size();a++)
            {
                for(int b=0;b<ns.elements.size();b++)
                {
                    if(ds.elements[a]==ns.elements[b])
                    {   count++;}
                }
            }
            //cout<<"count= "<<count<<" ds= "<<ds.elements.size()<<"ns= "<<ns.elements.size()<<endl;
            if(count==ds.elements.size() && count==ns.elements.size())
            {   return true;}
            else
            {   return false;}//means current ds is different from previous ds which was used to train the network
        }
        else 
        {   return false;}//means current ds is different from previous ds which was used to train the network
    }

    vector<filtered_data> f_train_data,f_test_data;
    data_package_class test_data;

    void train(data_package_class* data_pack,bool network_avail_status,int train_test_predict)//there cannot be a case of invalid network and data without labels.
    {
        string str_for_body;
        datapack_analyzer(data_pack);//function checked!
        //filtering data according to labels
        filter(data_pack,train_test_predict);//f_train_data pack gets created here.
        if(network_avail_status==true)
        {
            if(ds_ns_have_same_labels()==true)//training by batching
            {
                if(train_test_predict==0)//only training
                {
                    str_for_body.clear();
                    str_for_body.append("\nnetwork save file found. So training by batching without testing.");
                    body_engine_communication_data_obj.add_message(str_for_body);
                    simplex_solver_data_entry_point(f_train_data,&ds,&network1);
                    save_network();
                    str_for_body.clear();
                    str_for_body.append("\nnetwork saved");
                    body_engine_communication_data_obj.add_message(str_for_body);
                }
                else if(train_test_predict==1)//training with testing 
                {
                    str_for_body.clear();
                    str_for_body.append("\nnetwork save file found. So training by batching with testing.");
                    body_engine_communication_data_obj.add_message(str_for_body);
                    simplex_solver_data_entry_point(f_train_data,&ds,&network1);
                    save_network();
                    str_for_body.clear();
                    str_for_body.append("\nnetwork saved");
                    body_engine_communication_data_obj.add_message(str_for_body);
                    test();
                }
            }
            else
            {
                 //1. new variable is added to the data. Or adding new label.
                str_for_body.clear();
                str_for_body.append("ERROR!!!  The currently loaded network file is not associated with the currently loaded data \nfrom file you have selected as input source.");
                body_engine_communication_data_obj.add_message(str_for_body);
            }
        }
        else if(network_avail_status==false)
        {
            if(train_test_predict==0)//only training
            {
                str_for_body.clear();
                str_for_body.append("\nnetwork save file not found. Brand new training without testing.");
                body_engine_communication_data_obj.add_message(str_for_body);
                network_structure_modifier();
                simplex_solver_data_entry_point(f_train_data,&ds,&network1);
                save_network();
                str_for_body.clear();
                str_for_body.append("\nnetwork saved");
                body_engine_communication_data_obj.add_message(str_for_body);
            }
            else if(train_test_predict==1)//training with testing 
            {
                str_for_body.clear();
                str_for_body.append("\nnetwork save file not found. Brand new training with testing.");
                body_engine_communication_data_obj.add_message(str_for_body);
                network_structure_modifier();
                simplex_solver_data_entry_point(f_train_data,&ds,&network1);
                save_network();
                str_for_body.clear();
                str_for_body.append("\nnetwork saved");
                body_engine_communication_data_obj.add_message(str_for_body);
                test();
            }
        }
    }

    void test()//parameters not required now
    {
        string str_for_body;
        str_for_body.clear();
        str_for_body.append("\nTESTING PHASE");
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"\nTESTING PHASE\n"<<endl;
        // format is yes and then no naswers.
        ofstream file1("weight_matrix.csv",ios::out);
        file1<<"path id,output_neuron_id,weight matrix: ,";
        for(int b=0;b<network1.path[0].input_neuron_id.size();b++)
        {   file1<<network1.path[0].input_neuron_id[b]<<":";}
        file1<<"\n";
        for(int a=0;a<network1.path.size();a++)
        {
            file1<<network1.path[a].path_id<<",";
            file1<<network1.path[a].output_neuron_id<<":,";
            for(int b=0;b<network1.path[a].weight_matrix.size();b++)
            {   file1<<network1.path[a].weight_matrix[b]<<";";}
            file1<<"\n";
        }
        file1.close();
        str_for_body.clear();
        str_for_body.append("weight matrix saved\n");
        body_engine_communication_data_obj.add_message(str_for_body);
        cout<<"weight matrix saved\n";
        //entering data to network.
        for(int a=0;a<test_data.data.size();a++)
        {
            network1.enter_data_in_the_network(test_data.data[a]);
            network1.enter_label_neuron_to_be_fired(test_data.labels[a],ds.elements);
            network1.propagate();
        }
        network1.return_accuracy_details(body_engine_communication_data_obj.correct,
                                         body_engine_communication_data_obj.incorrect,
                                         body_engine_communication_data_obj.total,
                                         body_engine_communication_data_obj.double_fire,
                                         body_engine_communication_data_obj.not_all_fired,
                                         body_engine_communication_data_obj.wronglyfired,
                                         body_engine_communication_data_obj.accuracy);

        str_for_body.clear();
        str_for_body.append("\n\nRESULTS:\n");
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\ncorrect= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.correct));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\nincorrect= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.incorrect));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\ntotal= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.total));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\ndouble fired= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.double_fire));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\nnot at all fired= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.not_all_fired));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\nwrongly fired= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.wronglyfired));
        body_engine_communication_data_obj.add_message(str_for_body);

        str_for_body.clear();
        str_for_body.append("\n\nAccuracy= ");
        body_engine_communication_data_obj.add_message(str_for_body);
        str_for_body.clear();
        str_for_body.append(to_string(body_engine_communication_data_obj.accuracy));
        str_for_body.append(" %");
        body_engine_communication_data_obj.add_message(str_for_body);

        cout<<"accuracy= "<<network1.return_accuracy()<<endl;
    }

    void testing_for_each_label(data_package_class* data_pack/*,int train_test_predict*/)
    {
        datapack_analyzer(data_pack);
        filtered_data f_data;
        vector<filtered_data> f_data_vector;
        for(int a=0;a<ds.elements.size();a++)
        {
            f_data.data.clear();
            for(int b=0;b<data_pack->labels.size();b++)
            {
                if(ds.elements[a]==data_pack->labels[b])
                {
                    f_data.data.push_back(data_pack->data[b]);
                    f_data.label=data_pack->labels[b];
                }
            }
            f_data_vector.push_back(f_data);
        }
        //copying the data to f_train_data and f_test_data
        for(int a=0;a<f_data_vector.size();a++)
        {
            filtered_data f_data_temp;
            //copying data to f_test_data
            for(int b=0;b<f_data_vector[a].data.size();b++)
            {
                f_data_temp.data.push_back(f_data_vector[a].data[b]);
            }
            f_data_temp.label=f_data_vector[a].label;
            f_test_data.push_back(f_data_temp);
            f_data_temp.data.clear();
        }

        cout<<"\nTESTING FOR EACH LABELS: \n\n";
        for(int a=0;a<f_test_data.size();a++)
        {
            cout<<"\ntesting for data of label "<<f_test_data[a].label<<": ";
            network1.reset_statistics();
            for(int b=0;b<f_test_data[a].data.size();b++)
            {
                network1.enter_data_in_the_network(f_test_data[a].data[b]);
                network1.enter_label_neuron_to_be_fired(f_test_data[a].label,ds.elements);
                network1.propagate();
            }
            cout<<"accuracy= "<<network1.return_accuracy()<<endl;
        }
    }
    
    /*void predict_progress_bar()
    {
        while(shared_block_data_obj.predict_progress_bar_numerator<shared_block_data_obj.predict_progress_bar_denominator)
        {
            float x=shared_block_data_obj.predict_progress_bar_numerator,y=shared_block_data_obj.predict_progress_bar_denominator;
            if(y!=0)
            {
                struct winsize w;
                ioctl(0,TIOCGWINSZ,&w);
                float percentage=(x/y)*100;   
                system("clear");
                cout<<"\nprogress: ";
                float hl=w.ws_col/2;
                float ratio=100/hl;
                float pl=percentage*hl/100;
                for(int a=0;a<hl;a++)
                {
                    if(a<pl)
                    {   cout<<"#";}
                    else
                    {   cout<<".";}
                }
                cout<<"  "<<percentage<<"%";
                cout<<"  "<<shared_block_data_obj.predict_progress_bar_numerator<<" out of "<<shared_block_data_obj.predict_progress_bar_denominator<<" predictions complete"<<endl<<endl;
                sleep(1);
            }
        }
    }*/

    void predict(data_package_class* data_pack)
    {
        cout<<"\nprediction going on......";
        ofstream out_stream("prediction_result.csv",ios::out);
        out_stream<<"data,label,\n";
        network1.predict_mode=true;
        float label;
        shared_block_data_obj.predict_progress_bar_denominator=data_pack->data.size();
        
        //pthread_t predict_progress_bar_thread;
        //int predict_progress_bar_error;
        //if(pds==true)
        //{   predict_progress_bar_error=pthread_create(&predict_progress_bar_thread,NULL,(THREADFUNCPTR) &core_class::predict_progress_bar,NULL);}
        
        for(int a=0;a<data_pack->data.size();a++)
        {
            for(int b=0;b<data_pack->data[a].size();b++)
            {   out_stream<<data_pack->data[a][b]<<",";}
            network1.enter_data_in_the_network(data_pack->data[a]);
            network1.reset_all_output_neurons();
            //network1.enter_label_neuron_to_be_fired(data_pack->labels[a],network1.elements);
            network1.propagate();
            int fired_output_neuron_index=0;
            for(int b=0;b<network1.output_neurons.size();b++)
            {
                bool status=network1.output_neurons[b].return_fire_status();
                if(status==true)
                {   fired_output_neuron_index=b;}
            }
            label=network1.return_label_for_firing_neuron_index(fired_output_neuron_index);
            out_stream<<":"<<label<<",\n";
            shared_block_data_obj.predict_progress_bar_numerator++;
        }
        //if(pds==true)
        //{   predict_progress_bar_error=pthread_join(predict_progress_bar_thread,NULL);}
        //cout<<"accuracy= "<<network1.return_accuracy()<<endl;
        out_stream.close();
        cout<<"\nPrediction complete, check the file prediction_result.csv\n";
    }

    void make_prediction_on_user_entered_data()//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2@@@@@@@@@2
    {
        //function entering signal
        //input data size
        body_engine_communication_data_obj.size_of_input_data=network1.input_neuron_size();
        body_engine_communication_data_obj.prediction_on_individual_data_function_reached=true;
        float label;
        string str_to_body;
        while(body_engine_communication_data_obj.task_complete==false)
        {
            //waiting for data
            while(body_engine_communication_data_obj.data_entered==false)
            {
                Sleep(500);
                if(body_engine_communication_data_obj.task_complete==true)
                {   break;}
            }
            if(body_engine_communication_data_obj.task_complete==false)
            {
                //actual engine
                network1.enter_data_in_the_network(body_engine_communication_data_obj.input_data);
                network1.reset_all_output_neurons();
                //network1.enter_label_neuron_to_be_fired(data_pack->labels[a],network1.elements);
                network1.propagate();
                int fired_output_neuron_index=0;
                for(int b=0;b<network1.output_neurons.size();b++)
                {
                    bool status=network1.output_neurons[b].return_fire_status();
                    if(status==true)
                    {   fired_output_neuron_index=b;}
                }
                label=network1.return_label_for_firing_neuron_index(fired_output_neuron_index);
                //result
                body_engine_communication_data_obj.result=label;
                str_to_body.clear();
                str_to_body.append("\n\nResult for ");
                for(int a=0;a<network1.input_neuron_size();a++)
                {   str_to_body.append(to_string(body_engine_communication_data_obj.input_data[a]));
                    str_to_body.append(",");
                }
                str_to_body.append(" is ");
                str_to_body.append(to_string(label));
                body_engine_communication_data_obj.add_message(str_to_body);
            }
            body_engine_communication_data_obj.input_data.clear();
            //to enable the pause next time
            body_engine_communication_data_obj.data_entered=false;
            //function exiting signal
            body_engine_communication_data_obj_default.prediction_on_individual_data_function_reached=true;
        }
    }

    void set_the_no_of_threads_required(int no_of_c_datapacks_after_big_datapack_handling=0,bool tried_before=false)
    {
        //int no_of_physical_threads_in_cpu=sysconf(_SC_NPROCESSORS_ONLN);// for linux
        int no_of_physical_threads_in_cpu=thread::hardware_concurrency();// for windows
        SYSTEM_POWER_STATUS status;
        GetSystemPowerStatus(&status);
        int battery_available=(int)status.BatteryFlag;
        int prefered_no_of_threads=no_of_physical_threads_in_cpu+11;//for good speed in i5 7200

        string str1;
        str1.clear();
        str1.append("\nwaiting for user input (set no of threads)....");
        body_engine_communication_data_obj.add_message(str1);
        cout<<"\nwaiting for user input (set no of threads)....";
        body_engine_communication_data_obj.thread_selection_point_reached=true;
        while(body_engine_communication_data_obj.no_of_threads_setting_mode==false)
        {   Sleep(500);}
        if(body_engine_communication_data_obj.thread_set_automatic==true)
        {
            if(no_of_c_datapacks_after_big_datapack_handling>prefered_no_of_threads)
            {   required_no_of_threads=prefered_no_of_threads;}
            else
            {   required_no_of_threads=2;}
        }
        else
        {   required_no_of_threads=body_engine_communication_data_obj.set_no_of_threads;}
        string str_for_body;
        if(battery_available==1 && no_of_physical_threads_in_cpu<=4 && required_no_of_threads>4)
        {
            required_no_of_threads=no_of_physical_threads_in_cpu;
            str_for_body.clear();
            str_for_body.append("\nyour pc is not powerful enough so no of threads set to ");
            str_for_body.append(to_string(required_no_of_threads));
            str_for_body.append("\nthis process may take a lot of time....");
            body_engine_communication_data_obj.add_message(str_for_body);
        }

    }

    class body_engine_communication_data
    {
        private:
        bool message_ready=false;
        vector<string> message;
        //vector<string> message;
        bool data_ready_for_reading=false;
        unsigned int message_index=0;
        int intdata;
        bool intdata_ready=false;
        char char_data;
        bool char_data_ready=false;
        float floatdata;
        bool floatdata_ready=false;
        public:
        //make prediction on individual data
        bool data_entered=false;
        int size_of_input_data=0;
        float result;
        vector<float> input_data;
        bool prediction_on_individual_data_function_reached=false;
        //progress bar settings
        bool progress_bar_display=false;
        int progress_bar_value=0;

        //entire work complete or not
        bool task_complete=false;
        //thread seter
        bool no_of_threads_setting_mode=false;//stops the engine for the user entering the no of threads
        bool thread_selection_point_reached=false;//notifies the body that the engine has reached the thread setting point
        bool thread_set_automatic=false;
        int set_no_of_threads=1;
        //data
        float accuracy=0;
        int correct=0,incorrect=0,total=0,double_fire=0,not_all_fired=0,wronglyfired=0;

        string display_message()
        {
            if(message_ready==true && message_index<message.size())
            {
                string str=message[message_index];
                message_index++;
                return str;
            }
            else
            {   return "";}
        }

        void add_message(string str)
        {
            message_ready=false;
            message.push_back(str);
            message_ready=true;
        }

        void remove_message()
        {
            message_ready=false;
            message.pop_back();
            message_ready=true;
        }

        void send_data(int x)
        {
            intdata_ready=false;
            intdata=x;
            intdata_ready=true;
        }

        void send_data(char ch)
        {
            char_data_ready=false;
            char_data=ch;
            char_data_ready=true;
        }

        void send_data(float f)
        {
            floatdata_ready=false;
            floatdata=f;
            floatdata_ready=true;
        }

        void enable_reading_of_data(bool b)
        {   data_ready_for_reading=b;}

        int get_int_data()
        {
            if(data_ready_for_reading==true)
            {   return intdata;}
        }

        char get_char_data()
        {
            if(data_ready_for_reading==true)
            {   return char_data;}
        }

        float get_float_data()
        {
            if(data_ready_for_reading==true)
            {   return floatdata;}
        }
    };
    body_engine_communication_data body_engine_communication_data_obj_default;
    public:   

    void get_shared_block_data(int &datapacks_complete,int &total_c_datapacks,int &predict_numerator,int &predict_denominator)
    {
        datapacks_complete=shared_block_data_obj.no_of_c_datapacks_completed;
        total_c_datapacks=shared_block_data_obj.total_c_datapacks;
        predict_numerator=shared_block_data_obj.predict_progress_bar_numerator;
        predict_denominator=shared_block_data_obj.predict_progress_bar_denominator;
    }

    void restore_body_engine_communication_data_obj()
    {
        body_engine_communication_data_obj.accuracy=body_engine_communication_data_obj_default.accuracy;
        body_engine_communication_data_obj.task_complete=body_engine_communication_data_obj_default.task_complete;
        body_engine_communication_data_obj.set_no_of_threads=body_engine_communication_data_obj_default.set_no_of_threads;
        body_engine_communication_data_obj.thread_set_automatic=body_engine_communication_data_obj_default.thread_set_automatic;
        body_engine_communication_data_obj.no_of_threads_setting_mode=body_engine_communication_data_obj_default.no_of_threads_setting_mode;
        body_engine_communication_data_obj.thread_selection_point_reached=body_engine_communication_data_obj_default.thread_selection_point_reached;
        body_engine_communication_data_obj.total=body_engine_communication_data_obj_default.total;
        body_engine_communication_data_obj.correct=body_engine_communication_data_obj_default.correct;
        body_engine_communication_data_obj.accuracy=body_engine_communication_data_obj_default.accuracy;
        body_engine_communication_data_obj.incorrect=body_engine_communication_data_obj_default.incorrect;
        body_engine_communication_data_obj.double_fire=body_engine_communication_data_obj_default.double_fire;
        body_engine_communication_data_obj.wronglyfired=body_engine_communication_data_obj_default.wronglyfired;
        body_engine_communication_data_obj.not_all_fired=body_engine_communication_data_obj_default.not_all_fired;
        body_engine_communication_data_obj.progress_bar_display=body_engine_communication_data_obj_default.progress_bar_display;
        body_engine_communication_data_obj.progress_bar_value=body_engine_communication_data_obj_default.progress_bar_display;
        body_engine_communication_data_obj.data_entered=body_engine_communication_data_obj_default.data_entered;
        body_engine_communication_data_obj.size_of_input_data=body_engine_communication_data_obj_default.size_of_input_data;
        body_engine_communication_data_obj.prediction_on_individual_data_function_reached=body_engine_communication_data_obj_default.prediction_on_individual_data_function_reached;
    }
    body_engine_communication_data body_engine_communication_data_obj;

    void core_data_entry_point(data_package_class* data_pack,int train_test_predict,float &data_division1,string &network_save_file_name)//train_test_predict=1//train_test_predic is required for extra assurance
    {
        string str_for_body;
        if(train_test_predict==1)
        {   data_division=data_division1;}
        if(check_if_datapack_has_valid_labels(data_pack)==true && train_test_predict<2)//true
        {
            bool network_avail_status=load_network_if_available(0,0,false,"");
            train(data_pack,network_avail_status,train_test_predict);
        }
        else if(train_test_predict==3)//making prediction on used entered individual data.
        {
            bool network_load_status=load_network_if_available(0,0,true,network_save_file_name);            
            if(network_load_status==true)
            {   
                str_for_body.clear();
                str_for_body.append("Network successfully loaded");
                body_engine_communication_data_obj.add_message(str_for_body);
                make_prediction_on_user_entered_data();
            }
            else
            {
                str_for_body.clear();
                str_for_body.append("ERROR!!! failed to load network from the network file.");
                body_engine_communication_data_obj.add_message(str_for_body);
            }
        }
        else
        {
            if(load_network_if_available(0,0,true,network_save_file_name)==true && train_test_predict==2)//network avail,vaild label not avail,predic mode
            {   
                str_for_body.clear();
                str_for_body.append("Prediction going on....");
                body_engine_communication_data_obj.add_message(str_for_body);
                predict(data_pack);
                str_for_body.clear();
                str_for_body.append("\nprediction results daved in prediction_result.csv file");
                body_engine_communication_data_obj.add_message(str_for_body);
                //testing_for_each_label(data_pack);
            }
            else
            {   cout<<"Invalid option"<<endl;}//ERROR!!
        }
        Sleep(1000);//to let the progress display system catch up
        body_engine_communication_data_obj.task_complete=true;
        //restore_body_engine_communication_data_obj();
        cout<<"ending...";
    }
};
