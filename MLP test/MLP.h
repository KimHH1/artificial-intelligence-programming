#pragma once

#define LEARNING_RATE 0.1

class CMLP
{
public:
	CMLP();
	~CMLP();

		// �Ű�� ���������� ���� ����
		int m_iNumlnNodes;
		int m_iNumOutNodes;
		int m_iNumHiddenLayer;	//���緹�̾��� ��(hidden only)
		int m_iNumTotalLayer;	//��ü���̾��� ��(inputlayer+hiddenlayer+outputlayer)
		int* m_NumNodes;		//[0]-inputnode,[1..]-hidden layer,[m_iNumHiddenLayer+1],output layer,����


		double*** m_Weight;		//[����layer][���۳��][������]
		double** m_NodeOut;		//[layer][node]
		
		double** m_ErrorGradient; //[layer][node]

		double* pInValue, *pOutValue;	//�Է·��̾�,��·��̾�
		double* pCorrectOutValue;		//���䷹�̾�
		bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void Initw();
public:
	void Forward();
private:
	double ActivationFunc(double weigthsum);
public:
	void BackPopagationLearning();
};
