#include "MLP.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
CMLP MultiLayer;

int main()
{
	printf("학번 : 20190814\n이름 : 김현호\n과제 : 전방향 다층신경망 학습\n\n학습 전 가중치\n");

	int hidden_node[1] = {2};
	int numofHiddenLayer = 1;
	MultiLayer.Create(2,hidden_node, 1, numofHiddenLayer);
	int layer, snode, enode, node;
	
	double x[4][2] = { {0,0},{0,1},{1,0},{1,1} };

	MultiLayer.m_Weight[0][0][1] = -0.8;
	MultiLayer.m_Weight[0][1][1] = 0.5;
	MultiLayer.m_Weight[0][2][1] = 0.4;

	MultiLayer.m_Weight[0][0][2] = 0.1;
	MultiLayer.m_Weight[0][1][2] = 0.9;
	MultiLayer.m_Weight[0][2][2] = 1.0;

	MultiLayer.m_Weight[1][0][1] = -0.3;
	MultiLayer.m_Weight[1][1][1] = -1.2;
	MultiLayer.m_Weight[1][2][1] = 1.1;

	MultiLayer.pInValue[1] = 1;
	MultiLayer.pInValue[2] = 1;

	MultiLayer.pCorrectOutValue[1] = 0;

	for (layer = 0; layer < MultiLayer.m_iNumTotalLayer - 1; layer++)
	{
		printf("layer=%d: ", layer);
		for (enode = 1; enode <= MultiLayer.m_NumNodes[layer + 1]; enode++)
		{
			printf("[%d]bias=", enode);
			for (snode = 0; snode <= MultiLayer.m_NumNodes[layer]; snode++)
			{
				printf("%0.4lf ",round(MultiLayer.m_Weight[layer][snode][enode]*10000)/10000);
			}
		}
		printf("\n");
	}

	MultiLayer.Forward();
	MultiLayer.BackPopagationLearning();

	printf("\n%lf %lf=%lf\n\n",MultiLayer.pInValue[1], MultiLayer.pInValue[1], MultiLayer.pOutValue[1]);
		

	//노드의 출력값
	printf("노드별 출력값\n");
	for (layer = 0; layer < MultiLayer.m_iNumTotalLayer; layer++)
	{
		printf("layer=[%d]: ", layer);
		for (node = 1; node <= MultiLayer.m_NumNodes[layer]; node++)
		{
			printf("%0.4lf ", round(MultiLayer.m_NodeOut[layer][node]*10000)/10000);
		}
		printf("\n");
	}
	printf("\n");
	//에러경사값
	printf("에러경사값\n\n");
	for (layer = 0; layer < MultiLayer.m_iNumTotalLayer; layer++)
	{
		printf("layer=[%d]: ", layer);
		for (node = 1; node <= MultiLayer.m_NumNodes[layer]; node++)
		{
			printf("%0.4lf ",round(MultiLayer.m_ErrorGradient[layer][node]*10000)/10000);
		}
		printf("\n");
	}
	printf("\n");
	//학습 후 가중치 출력
	printf("학습 후 가중치\n");
	for (layer = 0; layer < MultiLayer.m_iNumTotalLayer - 1; layer++)
	{
		printf("layer=%d: ", layer);
		for (enode = 1; enode <= MultiLayer.m_NumNodes[layer + 1]; enode++)
		{
			printf("[%d]bias=", enode);
			for (snode = 0; snode <= MultiLayer.m_NumNodes[layer]; snode++)
			{
				printf("%0.4lf ",round(MultiLayer.m_Weight[layer][snode][enode]*10000)/10000);
			}
		}
		printf("\n");
	}
	/*int n;
	for (n = 0; n < 4; n++)
	{
		//MultiLayer.pInValue[0] =1; //바이어스
		MultiLayer.pInValue[1] = x[n][0];
		MultiLayer.pInValue[2] = x[n][1];

		MultiLayer.Forward();

		printf("%lf%lf=%lf\n", MultiLayer.pInValue[1], MultiLayer.pInValue[2], MultiLayer.pOutValue[1]);
	}
	printf("\n");
	*/

	/*가중치 출력 이게 원래 가중치 나오게 하는 거임 문제있으면 여기로
	for (layer = 0; layer < MultiLayer.m_iNumTotalLayer - 1; layer++)
	{
		for (snode = 0; snode <= MultiLayer.m_NumNodes[layer]; snode++)
		{
			for (enode = 1; enode <= MultiLayer.m_NumNodes[layer + 1]; enode++)
			{
				printf("w[%d][%d][%d] = %lf,", layer, snode, enode, MultiLayer.m_Weight[layer][snode][enode]);
			}
			printf("\n");
		}
		printf("\n");
	}
	*/

	return 0;
}


