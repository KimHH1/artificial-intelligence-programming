#include "MLP.h"
#include <stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

CMLP::CMLP()
{
	m_iNumlnNodes = 0;
	m_iNumOutNodes = 0;
	m_NumNodes = NULL;

	m_Weight = NULL;
	m_NodeOut = NULL;
	m_ErrorGradient = NULL;

	pInValue = NULL;
	pOutValue = NULL;
	pCorrectOutValue = NULL;
}

CMLP::~CMLP()
{
	int layer, snode, enode;
	
	if (m_NodeOut != NULL)
	{
		for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
			free(m_NodeOut[layer]);
		free(m_NodeOut);
	}
	if (m_Weight != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++)
		{
			if (m_Weight[layer] != NULL)
			{
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
					free(m_Weight[layer][snode]);
				free(m_Weight[layer]);
			}
		}
		free(m_Weight);
	}
	if(m_ErrorGradient != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer); layer++)
			free(m_ErrorGradient[layer]);
		free(m_ErrorGradient);
	}

	if (m_NumNodes != NULL)
		free(m_NumNodes);
}

bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer)
{
	int layer, snode, enode;

	m_iNumlnNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = numHiddenLayer;		//�Է�,����� ����
	m_iNumTotalLayer = numHiddenLayer + 2;	//���� + �Է� + ���

	//m_iNunmNodes�� ���� �޸��Ҵ�
	m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));	//������ �����ϱ����� +1��

	m_NumNodes[0] = m_iNumlnNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[1 + layer] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;	//����� ����
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;		//���� ����

	//m_NodeOut�� ���� �޸� �Ҵ�-[layerno][nodeno]
	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));		//����(+1)
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));	//���̾�� ���� +1

	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer - 1] + 1) * sizeof(double));		// ����(+1)

	//m_Weight�� ���� �޸� �Ҵ� - [����layer][���۳��][������]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));	//���̾ +1
		for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));		//�������� ���� +1 ���̾�� ���� +1

	}
	
	pInValue = m_NodeOut[0];
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

	Initw();
	//���̾�� ���� ��°� = 1

	for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
	{
		m_NodeOut[layer][0] = 1;
	}



	return false;
}


void CMLP::Initw()
{
	int layer, snode, enode;

	srand(time(NULL));

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (snode = 0; snode <= m_NumNodes[layer]; snode++)		//for ���̾�� ���� 0����
		{
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)	//���� ���̾��� ����
			{
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;
			}
		}
	}
}


void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			wsum = 0.0;
			wsum += m_Weight[layer][0][enode] * 1;
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
			}
			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
		
	}
}


double CMLP::ActivationFunc(double weigthsum)
{
	/*step func
	if (weigthsum > 0) return 1.0;
	else			   return 0.0;
	*/

	//sigmoid func
	return 1.0/(1.0 + exp(-weigthsum));
}


void CMLP::BackPopagationLearning()
{
	int layer,snode,enode;
	//������縦 ���� �޸� �Ҵ�
	if (m_ErrorGradient == NULL)
	{
		m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer]+1) * sizeof(double));
	}

	//����� ������� ���
	for (snode = 1; snode <= m_iNumOutNodes; snode++)
	{
		m_ErrorGradient[m_iNumTotalLayer - 1][snode] =
			(pCorrectOutValue[snode] - pOutValue[snode])
			*pOutValue[snode] * (1 - pOutValue[snode]);
	}

	//�߰��� ���� ��簪 ���
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (snode = 1; snode <= m_NumNodes[layer]; snode++)
		{
			m_ErrorGradient[layer][snode] = 0.0;
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
			{
				m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
			}
			m_ErrorGradient[layer][snode] *= (m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]));
		}
	}
	//����ġ ����
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode ++)
		{
			m_Weight[layer][0][enode] += (LEARNING_RATE*m_ErrorGradient[layer + 1][enode]*1);
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				m_Weight[layer][snode][enode] += (LEARNING_RATE*m_ErrorGradient[layer + 1][enode]*m_NodeOut[layer][snode]);
			}
		}
	}
}
