#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits.h>
#include <climits>
#include <vector>
#include <string.h>
#include <math.h>
#include <queue>
#include <list>
#include <map>
#define __STDC_LIMIT_MACROS
#include <stdint.h> 

using namespace cv;
using namespace std;

struct Edge
{   int v ;   
    int flow ; // flow of data in edge
    int capacity;    // capacity
    int rev ;    
	int pixelX;   // x cordinate of the starting pixel
	int pixelY;	  // Y cordinate of the starting pixel  
};

class Graph{
    int nodes;
    int *level ; 
	vector< Edge > *adjacencyList;        //vector to store all edges

public:
    Graph(int n){
        this->nodes = n;
        adjacencyList = new vector<Edge>[n];
        level = new int[n];    
    }
 
    void addEdge(int u,int v,double capacity, int x, int y){
		Edge edge;
		edge.v= v;
		edge.flow = 0; 
		edge.capacity = capacity;
		edge.rev = adjacencyList[v].size();
		edge.pixelX = x;
		edge.pixelY = y;
 
        Edge reverseEdge;
		reverseEdge.v= u;
		reverseEdge.flow = 0; 
		reverseEdge.capacity = 0;
		reverseEdge.rev = adjacencyList[v].size();
		reverseEdge.pixelX = x;
		reverseEdge.pixelY = y + 1;

        adjacencyList[u].push_back(edge);
        adjacencyList[v].push_back(reverseEdge);  
    }
    void minCut(Mat &inImage, int source, int sink,Mat &out_image);
    void ObjectLabelling(Mat &inImage, int source, Mat &out_image, bool ObjectFlag);
    bool BFS(int source, int sink);
    int sendFlow(int source, int flow, int sink, int st[]);
    void DinicMaxflow(int source, int sink);
};

bool Graph::BFS(int source, int sink)
{   list< int > queue;
    queue.push_back(source);
 
    vector<Edge>::iterator itr ;
 
    for (int i = 0 ; i < nodes ; i++)
	  {  level[i] = -1;
	  }
    level[source] = 0;  
   
  while (!queue.empty())
    {   int currentNode = queue.front();
        queue.pop_front();
		if(level[sink] > 0)
          return true;	
	
        for (itr = adjacencyList[currentNode].begin(); itr != adjacencyList[currentNode].end(); itr++)
        {   Edge &edge = *itr;
            if (level[edge.v] < 0  && edge.flow < edge.capacity)
            {  level[edge.v] = level[currentNode] + 1;
               queue.push_back(edge.v);
            }
        }
    }
 
    if(level[sink] < 0)
        return false;	
	else
       return true;
}

int Graph::sendFlow(int u, int flow, int sink, int st[])
{   if (u == sink)
        return flow;
 
    for (  ; st[u] < adjacencyList[u].size(); st[u]++)
    {   Edge &edge = adjacencyList[u][st[u]]; 
                                     
        if (level[edge.v] == level[u]+1 && edge.flow < edge.capacity)
        {   int currentFlow = min(flow, edge.capacity - edge.flow);
            int tempFlow = sendFlow(edge.v, currentFlow, sink, st);
 
            if (tempFlow > 0)
            {   edge.flow += tempFlow;
                adjacencyList[edge.v][edge.rev].flow -= tempFlow;
                return tempFlow;
            }
        }
    }
 
    return 0;
}


void Graph::DinicMaxflow(int source, int sink)
{   int totalFlow = 0;  
 
    //BFS will keep on running and cutting edges until sink is not reachable from source
    while (BFS(source, sink))
    {   int *st = new int[nodes+1];
        while (int flow = sendFlow(source, INT_MAX, sink, st))
            totalFlow += flow;
    }
}

void Graph::ObjectLabelling(Mat &inImage, int source, Mat &out_image, bool ObjectFlag)
{   vector<Edge>::iterator itr;
    for (int i = 0 ; i < nodes ; i++)
	  { level[i] = -1; 
      }
   
    level[source] = 0;  // Level of source vertex
    list< int > queue;
    queue.push_back(source);
 
      Vec3b pixel;
	 		pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
	 		
    while (!queue.empty())
    {   int currentNode = queue.front();
        queue.pop_front();
        for (itr = adjacencyList[currentNode].begin(); itr != adjacencyList[currentNode].end(); itr++)
        {   Edge &edge = *itr;
            if (level[edge.v] < 0  && edge.flow < edge.capacity)
		    {   
   		        out_image.at<Vec3b>(edge.pixelX, edge.pixelY) = pixel;		       
                level[edge.v] = level[currentNode] + 1;
                queue.push_back(edge.v);
            }
        }
    }
} 


void Graph::minCut(Mat &inImage, int source, int sink, Mat &out_image)
{  
   DinicMaxflow(source,sink);
		Vec3b pixel;
	 
		pixel[0] = 255;
				pixel[1] = 0;
				pixel[2] = 0;
	  
		
	
   for(int i=0;i<inImage.rows;i++)
	{ for(int j=0;j<inImage.cols;j++)	
		{ out_image.at<Vec3b>(i, j) = pixel;			  			  
		}
	}	
 
   ObjectLabelling(inImage, source,out_image, true);     //To color object

   }

double findPixelEnergy(Mat& inImage,int i,int j)
{    Vec3b pixelNx, pixelNy;
	 Vec3b pixelPx, pixelPy;	 
     int rows = inImage.rows;
     int cols = inImage.cols;	 
	 
	 pixelNx  = inImage.at<Vec3b>(i, j);
	return ((pixelNx[0]+pixelNx[1]+pixelNx[2])/3) ;

}

double findPixelEnergyRightEdge(Mat& inImage,int i,int j)
{    Vec3b pixelNx, pixelNy;
	 Vec3b pixelPx, pixelPy;	 
     int rows = inImage.rows;
     int cols = inImage.cols;	 
	//pixel (x + 1, y) 	N-Next	 X gradient
     if(j!=cols-1)
	     pixelNx  = inImage.at<Vec3b>(i, j+1);
	else
	     pixelNx  = inImage.at<Vec3b>(i, 0);
	//pixel (x − 1, y)  X gradient
	
	pixelPx  = inImage.at<Vec3b>(i, j);         
	 					
			 				 				
	   double xGradient = pow(pixelNx[0] - pixelPx[0],2) + pow(pixelNx[1] - pixelPx[1],2) + pow(pixelNx[2] - pixelPx[2],2);

	   return xGradient;
	
}
double findPixelEnergyDownEdge(Mat& inImage,int i,int j)
{    Vec3b pixelNx, pixelNy;
	 Vec3b pixelPx, pixelPy;	 
     int rows = inImage.rows;
     int cols = inImage.cols;
					
	//pixel (x, y+1) 	N-Next	 Y gradient
	if(i!=rows-1)
         pixelNy  = inImage.at<Vec3b>(i+1, j);
    else
         pixelNy  = inImage.at<Vec3b>(0, j); 

	//pixel (x − 1, y)   Y gradient
         pixelPy  = inImage.at<Vec3b>(i, j);
   
	 double yGradient = sqrt(pow(pixelNy[0] - pixelPy[0],2) + pow(pixelNy[1] - pixelPy[1],2) + pow(pixelNy[2] - pixelPy[2],2));
				

}

double calProbDistribution(Mat& inImage,double r, double u, double sigma, double u2, double sigma2, int i, int j)
{  double res ;

	static const double K = 0.39904344223;  //   value of 1/ sqrt(2*pi)
     if(sigma !=0)
	 { double a = pow((r - u),2);// / sigma;
       return (K / sqrt(sigma)) * std::exp(-0.5d * (a/sigma));
	}
     else
	 { Vec3b  pixel  = inImage.at<Vec3b>(i, j);
       double a = (pixel[0]+pixel[1]+pixel[2])/3;
       a = pow((a - u2),2) ;
	     return K * std::exp(-0.5d * a); 
	 }

}

double calInterPixelWeight(Mat& inImage,double w1, double w2, int i, int j,int x, int y)
{ double sigmaR = 1;
  double sigmaW = 1;
  double r = 1.0;
  double d = w1-w2;
  return std::exp(-r/sigmaR)* std::exp( -d*d/sigmaW);
 }

int main( int argc, char** argv )
{
    if(argc!=4){
        cout<<"Usage: ../seg input_image initialization_file output_mask"<<endl;
        return -1;
    }
    
    // Load the input image
    Mat inImage;
    inImage = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);
   
    if(!inImage.data){
        cout<<"Could not load input image!!!"<<endl;
        return -1;
    }

    if(inImage.channels()!=3){
        cout<<"Image does not have 3 channels!!! "<<inImage.depth()<<endl;
        return -1;
    }
    
    // the output image
    Mat out_image = inImage.clone();
    
    ifstream f(argv[2]);
    if(!f){
        cout<<"Could not load initial mask file!!!"<<endl;
        return -1;
    }
    
    int cols = inImage.cols, rows = inImage.rows, n;
    
	cout<<"cols:"<<cols<<endl;
	cout<<"rows:"<<rows<<endl;
	
    f>>n;
	vector<vector<double> > energy;
	energy.resize(rows,vector<double>(cols));
    
	for(int i=0;i<rows;i++)
	  {for(int j=0;j<cols;j++)
		{ energy[i][j]=   findPixelEnergy(inImage, i,j);		  
	    }
	  }
	
	vector<vector<int> > seeds;
	seeds.resize(n,vector<int>(3));
	
	int x, y, t,objectSeedsCount = 0,backSeedsCount = 0;
	double objectSum = 0,backSum=0,objectEnergyMean=0,backEnergyMean=0, objectEnergyDiff =0, backEnergyDiff=0, objectEnergyVariance=0,backEnergyVariance=0;
    double objectSum2 = 0,backSum2=0,objectEnergyMean2=0,backEnergyMean2=0, objectEnergyDiff2 =0, backEnergyDiff2=0, objectEnergyVariance2=0,backEnergyVariance2=0;
    Vec3b px;
	// get the seed pixels
    for(int i=0;i<n;++i){
        f>>y>>x>>t;
        if(x<0 || x>=rows || y<0 || y>=cols){
            cout<<"Invalid pixel mask!"<<endl;
            return -1;
        }
   
		seeds[i][0] = x;
		seeds[i][1] = y;
		seeds[i][2] = t;
		
	  px =inImage.at<Vec3b>(x, y);	
       if(t == 1)
	   {objectSum += energy[x][y];
        objectSum2 += ((px[0]+px[1]+px[2])/3);
	    objectSeedsCount++;
		cout<<"y:"<<y<<" x: "<<x<<" obj energy[x][y]:"<<energy[x][y]<<endl; 	     
	   }
	   else   
	   { backSum += energy[x][y];
        backSum2 += ((px[0]+px[1]+px[2])/3);   
	     backSeedsCount++;
		 cout<<"y:"<<y<<" x: "<<x<<" obj energy[x][y]:"<<energy[x][y]<<endl;
	    }

    }
	
   objectEnergyMean = objectSum/objectSeedsCount;
   backEnergyMean = backSum/backSeedsCount;
   objectEnergyMean2 = objectSum2/objectSeedsCount;
   backEnergyMean2 = backSum2/backSeedsCount;
	
   // Find Variance 
	for(int i=0;i<n;++i){
     px =inImage.at<Vec3b>(seeds[i][0],seeds[i][1]);		
 	 if(seeds[i][2] == 1)
	 {  objectEnergyDiff += pow( energy[seeds[i][0]][seeds[i][1]]- objectEnergyMean,2);
	    objectEnergyDiff2 += pow(((px[0]+px[1]+px[2])/3)- objectEnergyMean2,2);
	 }
	 else   
	 { backEnergyDiff += pow(energy[seeds[i][0]][seeds[i][1]] - backEnergyMean,2);
	   backEnergyDiff2 += pow(((px[0]+px[1]+px[2])/3)- backEnergyMean2,2);
	  }
	}   

	objectEnergyVariance = objectEnergyDiff/(objectSeedsCount-1);
    backEnergyVariance = backEnergyDiff/(backSeedsCount-1);
	objectEnergyVariance2 = objectEnergyDiff2/(objectSeedsCount-1);
    backEnergyVariance2 = backEnergyDiff2/(backSeedsCount-1);

	
	cout<<"objectEnergyMean:"<<objectEnergyMean<<endl;
	cout<<"objectEnergyVariance:"<<objectEnergyVariance<<endl;
	cout<<"backEnergyMean:"<<backEnergyMean<<endl;
	cout<<"backEnergyVariance:"<<backEnergyVariance<<endl;	

	vector<vector<int> > ObjectPixel;
	ObjectPixel.resize(rows,vector<int>(cols));
		
	objectSeedsCount = 0;
    backSeedsCount = 0;	
    int node=0, objectNode =0 , m=0,p=0;		
    bool seedFlag;

   Graph g(rows*cols+2);     //rows*cols+1 is the source(Object) and rows*cols+2 is the sink(Background)
   cout<<"rows*cols+2:"<<(rows*cols+2)<<endl;

 //Finding T-links using Gaussian distribution
	double probDistributionObjectTmp, probDistributionBackTmp, interPixelWeight, gradient;
	   
	for(int i=0;i<rows;i++)
	{ for(int j=0;j<cols;j++)	
	    {     seedFlag = false;        
		      for(int k=0;k<n;++k){
		         if(seeds[k][0]==i && seeds[k][1]==j)
					{ if(seeds[k][2] == 1)
						{	g.addEdge(rows*cols,node,2, i, j);
					        g.addEdge(node, rows*cols+1,1, i , j);
					    }					 
						else   
						{g.addEdge(node, rows*cols+1,2, i , j);
					    g.addEdge(rows*cols,node,1, i, j);
					        
						}
					  seedFlag = true;
			          break;
		            }
		         }
	          if(!seedFlag)
	          {  probDistributionObjectTmp = calProbDistribution(inImage, energy[i][j],objectEnergyMean,objectEnergyVariance,objectEnergyMean2,objectEnergyVariance2,i,j);
	             probDistributionBackTmp =  calProbDistribution(inImage, energy[i][j],backEnergyMean, backEnergyVariance, backEnergyMean2, backEnergyVariance2,i,j);
	             probDistributionObjectTmp = probDistributionObjectTmp / (probDistributionObjectTmp + probDistributionBackTmp);
		         probDistributionBackTmp = probDistributionBackTmp / (probDistributionObjectTmp + probDistributionBackTmp);
         //        g.addEdge(rows*cols,node,probDistributionObjectTmp, i, j);
		  //       g.addEdge(node, rows*cols+1,probDistributionBackTmp, i , j);
		            	     
		        if(probDistributionObjectTmp > probDistributionBackTmp)
		           {  g.addEdge(rows*cols,node,2, i, j);
		              g.addEdge(node, rows*cols+1,1, i , j);
		              objectSeedsCount++;
			       }
		        else
		           {  g.addEdge(rows*cols, node, 1,i, j);
		              g.addEdge(node, rows*cols+1,2,i, j);	     		     
			          backSeedsCount++;
		           } 
		       }
			   if(j<cols-1)
                {  gradient = findPixelEnergyRightEdge(inImage,i,j);
                   if(gradient<=0)
		             {    m++;
		                  gradient = 2;
		             }
		           else
		             {	 p++;		
		                 gradient = 1;
		             }
		            g.addEdge(node, node+1,gradient,i,j);  //edge with the right neighbour
			//		g.addEdge(node+1, node,gradient,i,j+1);  //edge with the right neighbour
					
       	    }
		
		if(i<rows-1)
		 {gradient = findPixelEnergyDownEdge(inImage,i,j);
	      if(gradient<=0)
		  {   m++;
             gradient = 2;
		  }
		  else
		  { p++ ;
             gradient = 1;		  
		  }
		  g.addEdge(node, node+cols,gradient,i,j);  //edge with the down neighbour
	     // g.addEdge(node+cols, node,gradient,i+1,j);  //edge with the down neighbour
		 }	     
	    node++;		 		 
		}
	}
	
	cout<<"Diff is less than 0 for "<<m<<endl;
	cout<<"Diff is greater than 0 for "<<p<<endl;	

   cout<<"objectSeedsCount:"<<objectSeedsCount<<" backSeedsCount:"<<backSeedsCount<<endl;
 
   g.minCut(inImage,rows*cols,rows*cols+1,out_image);
	
   imwrite( argv[3], out_image);
    
    // also display them both
   
    namedWindow( "Original image", WINDOW_AUTOSIZE );
    namedWindow( "Show Marked Pixels", WINDOW_AUTOSIZE );
    imshow( "Original image", inImage );
    imshow( "Show Marked Pixels", out_image );
    waitKey(0); 
    return 0;
}
