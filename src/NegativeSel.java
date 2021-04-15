import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.mapred.lib.FieldSelectionMapReduce;


public class NegativeSel
 {	
	public static int detNumber = 500;
	public static int fieldNumber = 15;
	public static int trainRecordsNumber = 17373;
	public static int testRecordsNumber = 8731;
	public static double threshold = 1.5;
	public static double threshold2 = 1.5;
	
	public static final double[][] records = new double[trainRecordsNumber][fieldNumber+1];
	
	static double[][] detectors = new double[detNumber][fieldNumber];
	
	public static void main(String[] args) throws IOException 
	{	
		createDetectors();
		detectNonSelf();	
	}

	private static void detectNonSelf() throws IOException 
	{
		int TN, TP, FN, FP;
		TN = FN = TP = FP = 0;
		
		readRecords("test2");
		
		normalize(testRecordsNumber);
		
		for(int i=0; i<testRecordsNumber; i++)
		{
			int j;
			for(j=0; j<detNumber; j++)
			{
				double dist = calculateDistance(records[i], detectors[j]);
				if(dist < threshold2)
				{
					//is nonself
					if(records[i][fieldNumber] == 1)
						TP++;
					else
						FP++;
					break;
				}						
				
			}
			
			if(j == detNumber)
			{
				//is self
				if(records[i][fieldNumber] == 0)
					TN++;
				else
					FN++;
			}
		}
		
		//write confusion matrix in file
        BufferedWriter bw = new BufferedWriter(new FileWriter("./SerialConfusionMatrix", false));
        bw.write("TN = " + Integer.toString(TN) + "\r\n");
        bw.write("TP = " + Integer.toString(TP) + "\r\n");
        bw.write("FN = " + Integer.toString(FN) + "\r\n");
        bw.write("FP = " + Integer.toString(FP));
        
        bw.close();
	}

	private static void createDetectors() throws IOException 
	{
		long trainTimeStart = 0, trainTimeElapsed = 0;
		trainTimeStart = System.currentTimeMillis();	//start time
		
		readRecords("input2");
		
		int n = 0;
		while(true)
		{
			detectors[n] = RandomGenerate();
			int i;
			
			for(i=0; i<trainRecordsNumber; i++)
			{
				if(records[i][fieldNumber] == 0)
				{
					double dist = calculateDistance(records[i], detectors[n]);
				
					if(dist < threshold)
						break;
				}
			}
			
			if(i==trainRecordsNumber)
				n++;
			
			if(n == detNumber)
				break;
		}
		
		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;	//time taken to create detectors
		
		//write Elapsed time in file
        BufferedWriter bw = new BufferedWriter(new FileWriter("./time", false));
        bw.write("Elapsed time = " + Long.toString(trainTimeElapsed) + "\r\n");
        
        bw.close();
		
	}

	private static double calculateDistance(double[] record, double[] det) 
	{
		// sum the squares
		double sum = 0.0;
		for (int i = 0; i < fieldNumber; i++)
		{		    
			sum += Math.pow(record[i]-det[i], 2);
		}
		// square root

		double distance = Math.sqrt(sum);
		return distance;
	}

	private static void readRecords(String filename) throws IOException 
	{
		//read reocrds from file
        FileReader f = new FileReader("./" + filename);
        BufferedReader br = new BufferedReader(f);
        
        String line = br.readLine();
        int i = 0;
        
        while(line != null)
        {
        	String[] sarray = line.split(",");
        	for(int j=0; j<fieldNumber+1; j++)
        	{
        		records[i][j] = Double.valueOf(sarray[j]);
        	}
        	i++;
        	line = br.readLine();
        }
	}

	private static double[] RandomGenerate() 
	{
		double[] d = new double[fieldNumber];
		Random rand = new Random();
		
		for (int i = 0; i < fieldNumber; i++)
			{		
				// random value
				d[i] = rand.nextDouble();				
			}
		
		return d;
	}

	
	public static void normalize(int recordsNum)
	{
		double[] min = new double[fieldNumber];		
		double[] MAX = new double[fieldNumber];
		
		for(int i=0; i<fieldNumber; i++)
		{
			min[i] = Double.POSITIVE_INFINITY;
			MAX[i] = Double.NEGATIVE_INFINITY;
			
			for(int j=0; j<recordsNum; j++)
			{				
				if(records[j][i] < min[i])
					min[i] = records[j][i];
				
				if(records[j][i] > MAX[i])
					MAX[i] = records[j][i];
			}
		}
		
		for(int i=0; i<recordsNum; i++)
		{
			for(int j=0; j<fieldNumber; j++)
			{
				if(min[j] == MAX[j])
					records[i][j] = 0;
				else
					records[i][j] = Math.abs(records[i][j] - min[j])/(MAX[j]-min[j]);
			}
		}
	}
	
	
}//class

