import net.librec.common.LibrecException;
import net.librec.recommender.AbstractRecommender;

import java.util.*;


/**
 * Created by Zicun Hang on 2018/6/8.
 */
public class LFMRecommender extends AbstractRecommender {

    //存储所有的userID
    protected ArrayList<Integer> userList = new ArrayList<>();
    //存储所有的itemID
    protected ArrayList<Integer> itemList = new ArrayList<>();

    //去重后的userList
    protected ArrayList<Integer> userIDs;
    //去重后的itemList
    protected ArrayList<Integer> itemIDs;


    //矩阵U的二维数组表示
    protected double[][] arrayP;
    //矩阵V的二维数组表示
    protected double[][] arrayQ;
    //存储用户-物品集，用于模型训练。
    protected Map<Integer, Map<Integer, Integer>> userItemListForCal = new HashMap<>();

    //getUserNegativeItem方法中为了方便嵌套结构排序而创建的类，无特殊含义。
    class Bean{
        Bean(int id, int c){
            this.itemID = id;
            this.count = c;
        }
        int itemID;
        int count;
    }


    @Override
    protected void setup() throws LibrecException {
        super.setup();
        String s = getDataModel().getTrainDataSet().toString();
        String[] data = s.split("\\s+");
        for(int i = 3; i < data.length; i++){
            if (i % 3 == 0){
                userList.add(Integer.parseInt(data[i]) + 1);
            }
            if(i % 3 == 1){
                itemList.add(Integer.parseInt(data[i]) + 1);
            }
        }
    }

    @Override
    protected void trainModel() throws LibrecException {
        latentFactorModel(5, 10, 0.02, 0.01);
    }

    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        double p = lfmPredict(userIdx+1, itemIdx+1, 5);
        return p;
    }



    //classCount:隐类数量 phaseCount:迭代次数 alpha:步长 lambda:正则化参数
    protected void latentFactorModel(int classCount, int phaseCount, double alpha, double lambda){

        initModel(classCount);

        for (int step = 0; step < phaseCount; step++) {
            for (int userID : userItemListForCal.keySet()) {
                for (int itemID : userItemListForCal.get(userID).keySet()) {
                    double eui = userItemListForCal.get(userID).get(itemID) -
                            lfmPredict(userID, itemID, classCount);
                    for (int f = 0; f < classCount; f++) {
                        System.out.println("step "+ (step+1) +" user "+ userID +" classCount " + (f+1));
                        arrayP[userIDs.indexOf(userID)][f] += alpha * (eui * arrayQ[f][itemIDs.indexOf(itemID)] -
                                lambda * arrayP[userIDs.indexOf(userID)][f]);
                        arrayQ[f][itemIDs.indexOf(itemID)] += alpha * (eui * arrayP[userIDs.indexOf(userID)][f] -
                                lambda * arrayQ[f][itemIDs.indexOf(itemID)]);
                    }

                }

            }
            alpha *= 0.9;
        }

    }

    //classCount:隐类数量
    protected void initModel(int classCount){
        userIDs = new ArrayList<Integer>(new HashSet<Integer>(userList));
        itemIDs = new ArrayList<Integer>(new HashSet<Integer>(itemList));


        arrayP = new double[userIDs.size()][classCount];
        for (int i = 0; i < userIDs.size(); i++) {
            for (int j = 0; j < classCount; j++){
                arrayP[i][j] = Math.random();
            }

        }

        arrayQ = new double[classCount][itemIDs.size()];
        for (int i = 0; i < classCount; i++) {
            for (int j = 0; j < itemIDs.size(); j++){
                arrayQ[i][j] = Math.random();
            }

        }

        for (int j = 0; j < userIDs.size(); j++){
            Map<Integer, Integer> userItem = new HashMap<>();
            ArrayList<Integer> positiveItemList = getUserPositiveItem(userIDs.get(j));
            ArrayList<Integer> negativeItemList = getUserNegativeItem(positiveItemList);

            for (int i = 0; i < positiveItemList.size(); i++){
                userItem.put(positiveItemList.get(i), 1);
            }

            for (int i = 0; i < negativeItemList.size(); i++){
                userItem.put(negativeItemList.get(i), 0);
            }
//            for (int key : userItem.keySet()) {
//                System.out.println(key + ":" + userItem.get(key));
//            }
            userItemListForCal.put(userIDs.get(j), userItem);
        }


    }

    //userID:用户ID
    protected ArrayList<Integer> getUserPositiveItem(int userID) {
        ArrayList<Integer> positiveItemList = new ArrayList<>();

        for (int i = 0; i < userList.size(); i++){
            if (userList.get(i) == userID){
                positiveItemList.add(itemList.get(i));
            }
        }
        return positiveItemList;
    }

    //u:getUserPositiveItem函数返回的正反馈物品，避免重复运算
    protected ArrayList<Integer> getUserNegativeItem(ArrayList<Integer> u) {
        class SortByCount implements Comparator {
            public int compare(Object o1, Object o2) {
                Bean b1 = (Bean) o1;
                Bean b2 = (Bean) o2;
                if (b1.count < b2.count)
                    return 1;
                else if (b1.count == b2.count)
                    return 0;
                return -1;
            }
        }
        //负反馈物品
        ArrayList<Integer> negativeItemList = new ArrayList<>();
        //用户未评分的物品
        ArrayList<Integer> otherItemList = (ArrayList<Integer>)itemList.clone();
        //记录一个物品被其他用户评分的次数
        ArrayList<Integer> itemCountList = new ArrayList<>();
        //存储用户未评分物品和对应的评分次数
        ArrayList<Bean> series = new ArrayList<>();

        otherItemList.removeAll(u);
        otherItemList = new ArrayList<Integer>(new HashSet<Integer>(otherItemList));


        for (int i = 0; i < otherItemList.size(); i++){
            int count = 0;
            for (int j = 0; j < itemList.size(); j++){
                if (itemList.get(j).equals(otherItemList.get(i))){
                    count++;
                }
            }
            itemCountList.add(count);
        }

        for (int i = 0; i < itemCountList.size(); i++){

            Bean b = new Bean(otherItemList.get(i),itemCountList.get(i));
            series.add(b);
        }

        Collections.sort(series, new SortByCount());
        series = new ArrayList<>(series.subList(0, u.size()));

        for (int i = 0; i < series.size(); i++){
            negativeItemList.add(series.get(i).itemID);
        }

        return negativeItemList;
    }

    //userID:目标用户 itemID:目标物品 classCount:隐类数量
    protected double lfmPredict(int userID, int itemID, int classCount){
        double[][] newP = new double[1][classCount];

        if (userIDs.indexOf(userID) == -1) return -100.1;
        if (itemIDs.indexOf(itemID) == -1) return -100.1;
        for(int i = 0; i < classCount; i++){
            newP[0][i] = arrayP[userIDs.indexOf(userID)][i];
        }

        double r = 0;
        for(int i = 0; i < classCount; i++){
            r += newP[0][i] * arrayQ[i][itemIDs.indexOf(itemID)];
        }
        r = HSF(r);
        return r;

    }

    //x:兴趣度
    protected double HSF(double x){
        double y;
        y = 1.0 / (1 + Math.exp(-x));
        return y;

    }

}
