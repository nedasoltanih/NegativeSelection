import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Random;

import javax.xml.bind.annotation.adapters.NormalizedStringAdapter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class NegativeSelMapReduce
 {	
	public static int detNumber = 50;
	public static int fieldNumber = 15;
	public static int trainRecordsNumber = 17373;
	public static int testRecordsNumber = 8731;
	public static double threshold = 1.5;
	public static double threshold2 = 1.5;
	
	public static final double[][] records = new double[trainRecordsNumber][fieldNumber+1];
	
	static double[][] detectors = new double[2 * detNumber][fieldNumber];
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException 
	{	
		createDetectors();
		detectNonSelf();	
	}

	private static void detectNonSelf() throws IOException 
	{
	
		double[][] detectors_1 = new double[detNumber][fieldNumber];
		
		int TN, TP, FN, FP;
		TN = FN = TP = FP = 0;
		
		readRecords("test2");
		
		normalize(testRecordsNumber);
        //read the detectors from file
        FileReader f = new FileReader("./part-r-00000");
        BufferedReader br = new BufferedReader(f);
        
        String line = br.readLine();
        int i = 0;
        
        while(line != null)
        {
        	String[] sarray = line.split(",");
        	for(int j=0; j<fieldNumber; j++)
        	{
        		detectors_1[i][j] = Double.valueOf(sarray[j]);
        	}
        	i++;
        	line = br.readLine();
        }
		
		for(i=0; i<testRecordsNumber; i++)
		{
			int j;
			for(j=0; j<detNumber; j++)
			{
				double dist = calculateDistance(records[i], detectors_1[j]);
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
        BufferedWriter bw = new BufferedWriter(new FileWriter("./MapRedConfusionMatrix", false));
        bw.write("TN = " + Integer.toString(TN) + "\r\n");
        bw.write("TP = " + Integer.toString(TP) + "\r\n");
        bw.write("FN = " + Integer.toString(FN) + "\r\n");
        bw.write("FP = " + Integer.toString(FP));
        
        bw.close();
	}

	private static void createDetectors() throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException 
	{
		long trainTimeStart = 0, trainTimeElapsed = 0;
		trainTimeStart = System.currentTimeMillis();
		
		//readRecords("input");
		
 		for(int i=0; i<detNumber; i++)
			detectors[i] = RandomGenerate();
		
		FileOutputStream fos;
		ObjectOutputStream out;
		String filename;

		filename = "detectors.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(detectors);
		out.close();
		
		
		String inputUri = "hdfs:///user/root/FraudDetection/input/";
        String outputUri = "hdfs:///user/root/FraudDetection/output/";
        String cacheUri  = "hdfs:///user/root/FraudDetection/cache/";
        
        Process p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./detectors.ser " + cacheUri);
        p.waitFor();
        
       
        Configuration conf = new Configuration();
        Job job = new Job(conf);

        job.setJarByClass(NegativeSelMapReduce.class);
        job.setJobName("Create Detectors");
        
        FileInputFormat.addInputPath(job, new Path(inputUri));
        FileOutputFormat.setOutputPath(job, new Path(outputUri));

        job.setMapperClass(NegSelMapper.class);
        job.setReducerClass(NegSelReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
		
        DistributedCache.addCacheFile(new URI(cacheUri+"detectors.ser"), job.getConfiguration());
        job.getConfiguration().setInt("fieldNumber", fieldNumber);
        job.getConfiguration().setInt("detNumber", detNumber);
        job.getConfiguration().set("threshold", Double.toString(threshold));
        
        job.waitForCompletion(true);
        
        
        
		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
		//write Elapsed time in file
        BufferedWriter bw = new BufferedWriter(new FileWriter("./MapRedTime", false));
        bw.write("Elapsed time = " + Long.toString(trainTimeElapsed) + "\r\n");
        
        bw.close();
        
		p = Runtime.getRuntime().exec("hadoop fs -copyToLocal " + outputUri + "part-r-00000 ./");
		p.waitFor();
		
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
	
	
	static class NegSelMapper extends Mapper<LongWritable, Text, Text, IntWritable>
	{
		private int detNumber;
		private int fieldNumber;
		private double threshold; 
		private double[][] detectors;
		
		
		public void configure(Configuration job) throws IOException, ClassNotFoundException
		{		
		detNumber = Integer.parseInt(job.get("detNumber"));
		fieldNumber = Integer.parseInt(job.get("fieldNumber"));
		threshold = Double.valueOf(job.get("threshold"));
		
		Path[] cacheFiles = new Path[0];
        cacheFiles = DistributedCache.getLocalCacheFiles(job);
        FileInputStream fis = null;
        ObjectInputStream in = null;
        
        for (Path cacheFile : cacheFiles)
        {
        	fis = new FileInputStream(cacheFile.toString());
        	in = new ObjectInputStream(fis);
        	try {
				detectors = (double[][]) in.readObject();
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        	in.close();            
        }
        
		}

        
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
		{
			
            try {
                configure(context.getConfiguration());}
            catch (ClassNotFoundException ex) {
                throw new RuntimeException("Cannot call configure function: "+ex.getMessage(), ex);}
            
		
			
			String s = Text.decode(value.getBytes());

            String[] sarray = s.split(",");
            double[] inputRecord = new double[sarray.length];
            
            for(int i=0; i<fieldNumber+1; i++)
            {
            	inputRecord[i] = Double.valueOf(sarray[i]);
            }

            
            int[] matched = new int[2 * detNumber];
            int h = 0;
            
            for(int i=0; i<detNumber + h; i++)
            {          	
            	matched[i] = 0;
            	
				if(inputRecord[fieldNumber] == 0)
				{
					double dist = calculateDistance(inputRecord, detectors[i]);
				
					//System.out.print("dist= " + Double.toString(dist));
					
					if(dist < threshold)
					{
						matched[i] = 1;
						
						detectors[h+detNumber] = RandomGenerate();
						h++;
					}
					
					else
					{
						String det = "";				
		                for(int j=0; j<fieldNumber; j++)
		                {
		                    det += Double.toString(detectors[i][j]);
		                    det += ",";
		                }
		                                
						context.write(new Text(det), new IntWritable(matched[i]));						
					}
				}				
            }                         
		}
	}
	
	
	static class NegSelReducer extends Reducer<Text, IntWritable, Text, IntWritable>
	{
		public void reduce(Text key, Iterable<IntWritable> value, Context context) throws IOException, InterruptedException
		{
			int count = 0;			
            for (IntWritable values : value) 
            {
            	count += Integer.parseInt(values.toString());            		            		
            }
            
            if(count == 0)
            	context.write(key, new IntWritable(count));
		}
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

