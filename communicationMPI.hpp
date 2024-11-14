#include <mpi.h>
#include <array>
#include <iostream>

#pragma once

#include "blockGrid.hpp"
#include "solverSetup.hpp"


template<int DIM, typename T_data>
class CommunicatorMPI{
    public:

    CommunicatorMPI(const BlockGrid<DIM,T_data>& blockGrid):
        blockGrid_(blockGrid),
        nranks_(blockGrid.getNranks()),
        globalLocation_(blockGrid.getGlobalLocation()),
        origin_(blockGrid.getOrigin()),
        ds_(blockGrid.getDs()),
        guards_(blockGrid.getGuards()),
        indexLimitsData_(blockGrid.getIndexLimitsData()),
        nlocal_noguards_(blockGrid.getNlocalNoGuards()),
        nlocal_guards_(blockGrid.getNlocalGuards()),
        ntotlocal_guards_(blockGrid.getNtotLocalGuards()),
        hasBoundary_(blockGrid.getHasBoundary()),
        bcsType_(blockGrid.getBcsType()),
        hasCommunication_(blockGrid.getHasCommunication()),
        numCommunication_(blockGrid.getNumCommunication()),
        numElementsComm_(blockGrid.getNumElementsComm()),
        stride_j_(blockGrid.getNlocalGuards()[0]),
        stride_k_(blockGrid.getNlocalGuards()[0]*blockGrid.getNlocalGuards()[1])
    
    {
        requestsSend_ = new MPI_Request[numCommunication_];
        requestsRcv_ = new MPI_Request[numCommunication_];
        statusesSend_ = new MPI_Status[numCommunication_];
        statusesRcv_ = new MPI_Status[numCommunication_];

    }

    ~CommunicatorMPI()
    {
        delete[] requestsSend_;
        delete[] requestsRcv_;
        delete[] statusesSend_;
        delete[] statusesRcv_;
    }


    void operator()(T_data data[])
    {
        std::array<int,3>  globalLocationOther;
        std::array<int,6> indexLimitsComm;
        int i,j,k;

        if(numCommunication_>0)
        {
            int countRequestSend=0;
            int countRequestRcv=0;

            for(int dir=0; dir< DIM; dir++)
            {
                if(dir==0)
                {
                    int* send_indx = new int[numElementsComm_[dir]];
                    int* rcv_indx = new int[numElementsComm_[dir]];
                    int* block_length = new int[numElementsComm_[dir]];
                    std::fill(block_length, block_length + numElementsComm_[dir], 1);

                    if(hasCommunication_[2*dir])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir+1]=indexLimitsData_[2*dir] + guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]--;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        std::array<int,3> arrayIndexGuards;
                        int indxLoop=0;
                        for(k=indexLimitsComm[4];k<indexLimitsComm[5];k++)
                        {
                            for(j=indexLimitsComm[2];j<indexLimitsComm[3];j++)
                            {
                                for(i=indexLimitsComm[0];i<indexLimitsComm[1];i++)
                                {
                                    arrayIndexGuards[0]=i;
                                    arrayIndexGuards[1]=j;
                                    arrayIndexGuards[2]=k;
                                    arrayIndexGuards[dir]-=guards_[dir];
                                    send_indx[indxLoop]=i + stride_j_*j + stride_k_*k;
                                    rcv_indx[indxLoop]=arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2];
                                    indxLoop++;       
                                }
                            }
                        }
                        MPI_Datatype send_typeX;
                        MPI_Type_indexed(numElementsComm_[dir], block_length, send_indx, getMPIType<T_data>(), &send_typeX);
                        MPI_Type_commit(&send_typeX);
                        MPI_Datatype recv_typeX;
                        MPI_Type_indexed(numElementsComm_[dir], block_length, rcv_indx, getMPIType<T_data>(), &recv_typeX);
                        MPI_Type_commit(&recv_typeX);

                        MPI_Isend(data, 1, send_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(data, 1, recv_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }
                    if(hasCommunication_[2*dir+1])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir]=indexLimitsData_[2*dir+1] - guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]++;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        std::array<int,3> arrayIndexGuards;
                        int indxLoop=0;
                        for(k=indexLimitsComm[4];k<indexLimitsComm[5];k++)
                        {
                            for(j=indexLimitsComm[2];j<indexLimitsComm[3];j++)
                            {
                                for(i=indexLimitsComm[0];i<indexLimitsComm[1];i++)
                                {
                                    arrayIndexGuards[0]=i;
                                    arrayIndexGuards[1]=j;
                                    arrayIndexGuards[2]=k;
                                    arrayIndexGuards[dir]+=guards_[dir];
                                    send_indx[indxLoop]=i + stride_j_*j + stride_k_*k;
                                    rcv_indx[indxLoop]=arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2];
                                    indxLoop++;
                                }
                            }
                        }
                        MPI_Datatype send_typeX;
                        MPI_Type_indexed(numElementsComm_[dir], block_length, send_indx, getMPIType<T_data>(), &send_typeX);
                        MPI_Type_commit(&send_typeX);
                        MPI_Datatype recv_typeX;
                        MPI_Type_indexed(numElementsComm_[dir], block_length, rcv_indx, getMPIType<T_data>(), &recv_typeX);
                        MPI_Type_commit(&recv_typeX);

                        MPI_Isend(data, 1, send_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(data, 1, recv_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }

                    delete[] send_indx;
                    delete[] rcv_indx;
                    delete[] block_length;
                }
                else if(dir==1)
                {
                    int* send_indx = new int[nlocal_noguards_[dir+1]];
                    int* rcv_indx = new int[nlocal_noguards_[dir+1]];
                    int* block_length = new int[nlocal_noguards_[dir+1]];
                    std::fill(block_length, block_length + nlocal_noguards_[dir+1], nlocal_noguards_[dir-1]);

                    if(hasCommunication_[2*dir])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir+1]=indexLimitsData_[2*dir] + guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]--;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        std::array<int,3> arrayIndexGuards;
                        int indxLoop=0;
                        i=indexLimitsComm[0];
                        for(k=indexLimitsComm[4];k<indexLimitsComm[5];k++)
                        {
                            for(j=indexLimitsComm[2];j<indexLimitsComm[3];j++)
                            {
                                arrayIndexGuards[0]=i;
                                arrayIndexGuards[1]=j;
                                arrayIndexGuards[2]=k;
                                arrayIndexGuards[dir]-=guards_[dir];
                                send_indx[indxLoop]=i + stride_j_*j + stride_k_*k;
                                rcv_indx[indxLoop]=arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2];
                                indxLoop++;       
                            }
                        }
                        MPI_Datatype send_typeX;
                        MPI_Type_indexed(nlocal_noguards_[dir+1], block_length, send_indx, getMPIType<T_data>(), &send_typeX);
                        MPI_Type_commit(&send_typeX);
                        MPI_Datatype recv_typeX;
                        MPI_Type_indexed(nlocal_noguards_[dir+1], block_length, rcv_indx, getMPIType<T_data>(), &recv_typeX);
                        MPI_Type_commit(&recv_typeX);

                        MPI_Isend(data, 1, send_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(data, 1, recv_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }
                    if(hasCommunication_[2*dir+1])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir]=indexLimitsData_[2*dir+1] - guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]++;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        std::array<int,3> arrayIndexGuards;
                        int indxLoop=0;
                        i=indexLimitsComm[0];
                        for(k=indexLimitsComm[4];k<indexLimitsComm[5];k++)
                        {
                            for(j=indexLimitsComm[2];j<indexLimitsComm[3];j++)
                            {
                                arrayIndexGuards[0]=i;
                                arrayIndexGuards[1]=j;
                                arrayIndexGuards[2]=k;
                                arrayIndexGuards[dir]+=guards_[dir];
                                send_indx[indxLoop]=i + stride_j_*j + stride_k_*k;
                                rcv_indx[indxLoop]=arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2];
                                indxLoop++;
                            }
                        }
                        MPI_Datatype send_typeX;
                        MPI_Type_indexed(nlocal_noguards_[dir+1], block_length, send_indx, getMPIType<T_data>(), &send_typeX);
                        MPI_Type_commit(&send_typeX);
                        MPI_Datatype recv_typeX;
                        MPI_Type_indexed(nlocal_noguards_[dir+1], block_length, rcv_indx, getMPIType<T_data>(), &recv_typeX);
                        MPI_Type_commit(&recv_typeX);

                        MPI_Isend(data, 1, send_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(data, 1, recv_typeX, rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }

                    delete[] send_indx;
                    delete[] rcv_indx;
                    delete[] block_length;

                }
                else if(dir==2)
                {
                    if(hasCommunication_[2*dir])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir+1]=indexLimitsData_[2*dir] + guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]--;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        int numElementsCommZ = nlocal_guards_[0]*nlocal_noguards_[1] - guards_[0];
                        std::array<int,3> arrayIndexGuards;
                        k=indexLimitsComm[4];
                        j=indexLimitsComm[2];
                        i=indexLimitsComm[0];
                        arrayIndexGuards[0]=i;
                        arrayIndexGuards[1]=j;
                        arrayIndexGuards[2]=k;
                        arrayIndexGuards[dir]-=guards_[dir];
                        MPI_Isend(&data[i + stride_j_*j + stride_k_*k], numElementsCommZ, getMPIType<T_data>(), rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(&data[arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2]], numElementsCommZ, getMPIType<T_data>(), rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }
                    if(hasCommunication_[2*dir+1])
                    {
                        // set index boundary for cells in axis dir --> index limits refer to send data
                        indexLimitsComm = indexLimitsData_;
                        indexLimitsComm[2*dir]=indexLimitsData_[2*dir+1] - guards_[dir];
                        // get other rank location
                        globalLocationOther = globalLocation_;
                        globalLocationOther[dir]++;
                        int rankOther=globalLocationOther[0] + globalLocationOther[1]*nranks_[0] + globalLocationOther[2]*nranks_[0]*nranks_[1];
                        int numElementsCommZ = nlocal_guards_[0]*nlocal_noguards_[1] - guards_[0]; 
                        std::array<int,3> arrayIndexGuards;
                        k=indexLimitsComm[4];
                        j=indexLimitsComm[2];
                        i=indexLimitsComm[0];
                        arrayIndexGuards[0]=i;
                        arrayIndexGuards[1]=j;
                        arrayIndexGuards[2]=k;
                        arrayIndexGuards[dir]+=guards_[dir];
                        MPI_Isend(&data[i + stride_j_*j + stride_k_*k], numElementsCommZ, getMPIType<T_data>(), rankOther, 0, MPI_COMM_WORLD, &requestsSend_[countRequestSend]);
                        countRequestSend++;
                        MPI_Irecv(&data[arrayIndexGuards[0] + stride_j_*arrayIndexGuards[1] + stride_k_*arrayIndexGuards[2]], numElementsCommZ, getMPIType<T_data>(), rankOther, 0, MPI_COMM_WORLD, &requestsRcv_[countRequestRcv]);
                        countRequestRcv++;
                    }
                }
            }
        }
    }

    void waitAllandCheckSend() const
    {
        MPI_Waitall(numCommunication_, requestsSend_, statusesSend_);
        for (int i = 0; i < numCommunication_; i++) 
        {
            if (statusesSend_[i].MPI_ERROR != MPI_SUCCESS) 
            {
                std::cerr << "Error in MPI send " << i << ": " << statusesSend_[i].MPI_ERROR << std::endl;
            }
        }
    }

    void waitAllandCheckRcv() const 
    {
        MPI_Waitall(numCommunication_, requestsRcv_, statusesRcv_);
        for (int i = 0; i < numCommunication_; i++) 
        {
            if (statusesRcv_[i].MPI_ERROR != MPI_SUCCESS) 
            {
                std::cerr << "Error in MPI recv " << i << ": " << statusesRcv_[i].MPI_ERROR << std::endl;
            }
        }
    }                

    private:

    const BlockGrid<DIM,T_data>& blockGrid_;
    const std::array<int,3>  nranks_;
    const std::array<int,3> globalLocation_;
    const std::array<T_data,3> origin_;
    const std::array<T_data,3> ds_;
    const std::array<int,3> guards_;
    const std::array<int,6> indexLimitsData_;
    const std::array<int,3> nlocal_noguards_;
    const std::array<int,3> nlocal_guards_;
    const int ntotlocal_guards_;
    const std::array<bool,6> hasBoundary_;
    const std::array<int,6> bcsType_;
    const std::array<bool,6> hasCommunication_;
    const int numCommunication_;
    const std::array<int,3> numElementsComm_;
    const int stride_j_;
    const int stride_k_;
    

    MPI_Request* requestsSend_;
    MPI_Request* requestsRcv_;
    MPI_Status*  statusesSend_;
    MPI_Status*  statusesRcv_;

};