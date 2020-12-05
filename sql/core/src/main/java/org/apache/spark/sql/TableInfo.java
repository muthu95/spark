package org.apache.spark.sql;

import java.util.Set;

public class TableInfo {
    public String pKeyName;
    public Set<Object> set;

    public TableInfo(String pKeyName, Set<Object> set) {
        this.pKeyName = pKeyName;
        this.set = set;
    }
}