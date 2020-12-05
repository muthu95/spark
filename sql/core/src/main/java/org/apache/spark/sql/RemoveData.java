package org.apache.spark.sql;

import java.util.HashMap;

public class RemoveData {
    public static HashMap<String, TableInfo> removeMap = new HashMap<>();

    public static String extractPathKey(String path) {
        int eIdx = path.indexOf(".parquet");
        int sIdx = path.lastIndexOf('/', eIdx) + 1;
        return path.substring(sIdx, eIdx);
    }
}
