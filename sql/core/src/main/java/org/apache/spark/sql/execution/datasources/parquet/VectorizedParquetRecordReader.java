/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.datasources.parquet;

import java.io.IOException;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.io.InvalidRecordException;
import org.apache.parquet.schema.Type;

import org.apache.spark.sql.RemoveData;
import org.apache.spark.memory.MemoryMode;
import org.apache.spark.sql.TableInfo;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.vectorized.ColumnVectorUtils;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.vectorized.WritableColumnVector;
import org.apache.spark.sql.execution.vectorized.OffHeapColumnVector;
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A specialized RecordReader that reads into InternalRows or ColumnarBatches directly using the
 * Parquet column APIs. This is somewhat based on parquet-mr's ColumnReader.
 *
 * TODO: handle complex types, decimal requiring more than 8 bytes, INT96. Schema mismatch.
 * All of these can be handled efficiently and easily with codegen.
 *
 * This class can either return InternalRows or ColumnarBatches. With whole stage codegen
 * enabled, this class returns ColumnarBatches which offers significant performance gains.
 * TODO: make this always return ColumnarBatches.
 */
public class VectorizedParquetRecordReader extends SpecificParquetRecordReaderBase<Object> {

  // The capacity of vectorized batch.
  private int capacity;

  /**
   * Batch of rows that we assemble and the current index we've returned. Every time this
   * batch is used up (batchIdx == numBatched), we populated the batch.
   */
  private int batchIdx = 0;
  private int numBatched = 0;

  /**
   * For each request column, the reader to read this column. This is NULL if this column
   * is missing from the file, in which case we populate the attribute with NULL.
   */
  private VectorizedColumnReader[] columnReaders;

  /**
   * The number of rows that have been returned.
   */
  private long rowsReturned;

  /**
   * The number of rows that have been reading, including the current in flight row group.
   */
  private long totalCountLoadedSoFar = 0;

  /**
   * For each column, true if the column is missing in the file and we'll instead return NULLs.
   */
  private boolean[] missingColumns;

  /**
   * The timezone that timestamp INT96 values should be converted to. Null if no conversion. Here to
   * workaround incompatibilities between different engines when writing timestamp values.
   */
  private final ZoneId convertTz;

  /**
   * The mode of rebasing date/timestamp from Julian to Proleptic Gregorian calendar.
   */
  private final String datetimeRebaseMode;

  /**
   * The mode of rebasing INT96 timestamp from Julian to Proleptic Gregorian calendar.
   */
  private final String int96RebaseMode;

  /**
   * columnBatch object that is used for batch decoding. This is created on first use and triggers
   * batched decoding. It is not valid to interleave calls to the batched interface with the row
   * by row RecordReader APIs.
   * This is only enabled with additional flags for development. This is still a work in progress
   * and currently unsupported cases will fail with potentially difficult to
   * diagnose errors.
   * This should be only turned on for development to work on this feature.
   *
   * When this is set, the code will branch early on in the RecordReader APIs
   * . There is no shared
   * code between the path that uses the MR decoders and the vectorized ones.
   *
   * TODOs:
   *  - Implement v2 page formats (just make sure we create the correct
   *  decoders).
   */
  private ColumnarBatch columnarBatch;
    private ColumnarBatch columnarBatch2;

    private WritableColumnVector[] columnVectors;
    private WritableColumnVector[] columnVectors2;

  /**
   * If true, this class returns batches instead of rows.
   */
  private boolean returnColumnarBatch;

  /**
   * The memory mode of the columnarBatch
   */
  private final MemoryMode MEMORY_MODE;

  public VectorizedParquetRecordReader(
      ZoneId convertTz,
      String datetimeRebaseMode,
      String int96RebaseMode,
      boolean useOffHeap,
      int capacity) {
    this.convertTz = convertTz;
    this.datetimeRebaseMode = datetimeRebaseMode;
    this.int96RebaseMode = int96RebaseMode;
    MEMORY_MODE = useOffHeap ? MemoryMode.OFF_HEAP : MemoryMode.ON_HEAP;
    this.capacity = capacity;
  }

  // For test only.
  public VectorizedParquetRecordReader(boolean useOffHeap, int capacity) {
    this(null, "CORRECTED", "LEGACY", useOffHeap, capacity);
  }

  /**
   * Implementation of RecordReader API.
   */
  @Override
  public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext)
      throws IOException, InterruptedException, UnsupportedOperationException {
    super.initialize(inputSplit, taskAttemptContext);
    initializeInternal();
  }

  /**
   * Utility API that will read all the data in path. This circumvents the need to create Hadoop
   * objects to use this class. `columns` can contain the list of columns to project.
   */
  @Override
  public void initialize(String path, List<String> columns) throws IOException,
      UnsupportedOperationException {
    super.initialize(path, columns);
    initializeInternal();
  }

  @Override
  public void close() throws IOException {
    if (columnarBatch != null) {
      columnarBatch.close();
      columnarBatch = null;
    }
    super.close();
  }

  @Override
  public boolean nextKeyValue() throws IOException {
    resultBatch();

    if (returnColumnarBatch) return nextBatch();

    if (batchIdx >= numBatched) {
      if (!nextBatch()) return false;
    }
    ++batchIdx;
    return true;
  }

  @Override
  public Object getCurrentValue() {
      if (returnColumnarBatch) {
          return columnarBatch2;
      }
      return columnarBatch2.getRow(batchIdx - 1);
  }

  @Override
  public float getProgress() {
    return (float) rowsReturned / totalRowCount;
  }

  // Creates a columnar batch that includes the schema from the data files and the additional
  // partition columns appended to the end of the batch.
  // For example, if the data contains two columns, with 2 partition columns:
  // Columns 0,1: data columns
  // Column 2: partitionValues[0]
  // Column 3: partitionValues[1]
  private void initBatch(
      MemoryMode memMode,
      StructType partitionColumns,
      InternalRow partitionValues) {
    StructType batchSchema = new StructType();
    for (StructField f: sparkSchema.fields()) {
      batchSchema = batchSchema.add(f);
    }
    if (partitionColumns != null) {
      for (StructField f : partitionColumns.fields()) {
        batchSchema = batchSchema.add(f);
      }
    }

    if (memMode == MemoryMode.OFF_HEAP) {
        columnVectors =
                OffHeapColumnVector.allocateColumns(capacity, batchSchema);
        columnVectors2 = OffHeapColumnVector.allocateColumns(capacity,
                batchSchema);
    } else {
        columnVectors =
                OnHeapColumnVector.allocateColumns(capacity, batchSchema);
        columnVectors2 = OnHeapColumnVector.allocateColumns(capacity,
                batchSchema);
    }
      columnarBatch = new ColumnarBatch(columnVectors);
      columnarBatch2 = new ColumnarBatch(columnVectors2);
    if (partitionColumns != null) {
      int partitionIdx = sparkSchema.fields().length;
      for (int i = 0; i < partitionColumns.fields().length; i++) {
          ColumnVectorUtils.populate(columnVectors[i +
                  partitionIdx], partitionValues, i);
          columnVectors2[i + partitionIdx] = columnVectors[i + partitionIdx];
          columnVectors[i + partitionIdx].setIsConstant();
          columnVectors2[i + partitionIdx].setIsConstant();
      }
    }

    // Initialize missing columns with nulls.
    for (int i = 0; i < missingColumns.length; i++) {
      if (missingColumns[i]) {
          columnVectors[i].putNulls(0, capacity);
          columnVectors2[i].putNulls(0, capacity);
          columnVectors[i].setIsConstant();
          columnVectors2[i].setIsConstant();
      }
    }
  }

  private void initBatch() {
    initBatch(MEMORY_MODE, null, null);
  }

  public void initBatch(StructType partitionColumns, InternalRow partitionValues) {
    initBatch(MEMORY_MODE, partitionColumns, partitionValues);
  }

  /**
   * Returns the ColumnarBatch object that will be used for all rows returned by this reader.
   * This object is reused. Calling this enables the vectorized reader. This should be called
   * before any calls to nextKeyValue/nextBatch.
   */
  public ColumnarBatch resultBatch() {
    if (columnarBatch == null) initBatch();
    return columnarBatch;
  }

  /**
   * Can be called before any rows are returned to enable returning columnar
   * batches directly.
   */
  public void enableReturningBatches() {
    returnColumnarBatch = true;
  }

  public Object getPrimaryKeyValue(InternalRow currRow,
                                   DataType pKeyDataType,
                                   int pKeyOrdinal) {
    Object pKey = null;
    if (pKeyDataType == DataTypes.BooleanType) {
      pKey = currRow.getBoolean(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.ByteType) {
      pKey = currRow.getByte(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.ShortType) {
      pKey = currRow.getShort(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.IntegerType) {
      pKey = currRow.getInt(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.LongType) {
      pKey = currRow.getLong(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.FloatType) {
      pKey = currRow.getFloat(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.DoubleType) {
      pKey = currRow.getDouble(pKeyOrdinal);
    } else if (pKeyDataType == DataTypes.StringType) {
      pKey = currRow.getUTF8String(pKeyOrdinal).toString();
    } else {
      System.out.println("Muthu: INVALID DTYPE");
    }
    return pKey;
  }

  /* public void mergeRow(Map<String, Object> changes, int rowIdx) {
    if (changes == null) {
      return;
    }
    for (String colName : changes.keySet()) {
      try {
        int colIdx = requestedSchema.getFieldIndex(colName);
        DataType colType = columnVectors[colIdx].dataType();
        Object newObject = changes.get(colName);
        if (colType == DataTypes.BooleanType) {
          columnVectors[colIdx].putBoolean(rowIdx, (Boolean) newObject);
        } else if (colType == DataTypes.ByteType) {
          columnVectors[colIdx].putByte(rowIdx, (Byte) newObject);
        } else if (colType == DataTypes.ShortType) {
          columnVectors[colIdx].putShort(rowIdx, (Short) newObject);
        } else if (colType == DataTypes.IntegerType) {
          columnVectors[colIdx].putInt(rowIdx, (Integer) newObject);
        } else if (colType == DataTypes.LongType) {
          columnVectors[colIdx].putLong(rowIdx, (Long) newObject);
        } else if (colType == DataTypes.FloatType) {
          columnVectors[colIdx].putFloat(rowIdx, (Float) newObject);
        } else if (colType == DataTypes.DoubleType) {
          columnVectors[colIdx].putDouble(rowIdx, (Double) newObject);
        } else if (colType == DataTypes.StringType) {
          columnVectors[colIdx].putByteArray(rowIdx,
                  ((String) newObject).getBytes());
        } else {
          System.out.println("Muthu: INVALID DTYPE");
        }
      } catch (InvalidRecordException e) {
        System.out.println("Muthu: COLUMN TO MERGE NOT FOUND");
      }
    }
  } */

    public void includeRow(InternalRow row) {
        for (int fieldIdx = 0; fieldIdx < row.numFields(); fieldIdx++) {
            DataType t = columnVectors2[fieldIdx].dataType();
            WritableColumnVector col = columnVectors2[fieldIdx];
            if (row.isNullAt(fieldIdx)) {
                col.appendNull();
            } else {
                if (t == DataTypes.BooleanType) {
                    col.appendBoolean(row.getBoolean(fieldIdx));
                } else if (t == DataTypes.ByteType) {
                    col.appendByte(row.getByte(fieldIdx));
                } else if (t == DataTypes.ShortType) {
                    col.appendShort(row.getShort(fieldIdx));
                } else if (t == DataTypes.IntegerType) {
                    col.appendInt(row.getInt(fieldIdx));
                } else if (t == DataTypes.LongType) {
                    col.appendLong(row.getLong(fieldIdx));
                } else if (t == DataTypes.FloatType) {
                    col.appendFloat(row.getFloat(fieldIdx));
                } else if (t == DataTypes.DoubleType) {
                    col.appendDouble(row.getDouble(fieldIdx));
                } else if (t == DataTypes.StringType) {
                    UTF8String v = row.getUTF8String(fieldIdx);
                    byte[] bytes = v.getBytes();
                    col.appendByteArray(bytes, 0, bytes.length);
                } else {
                    System.out.println("Muthu: INVALID DTYPE");
                }
            }
        }
    }

    public boolean mergeBatch() {
        try {
            String pathKey = RemoveData.extractPathKey(file.toString());
            TableInfo tableInfo = RemoveData.removeMap.get(pathKey);
            int pKeyOrdinal = (tableInfo == null) ? -1 :
                    requestedSchema.getFieldIndex(tableInfo.pKeyName);
            Iterator<InternalRow> iter = columnarBatch.rowIterator();
            int rowIdx = 0;
            while (iter.hasNext()) {
                InternalRow currRow = iter.next();
                if (pKeyOrdinal == -1) {
                    includeRow(currRow);
                    rowIdx++;
                } else {
                    DataType pKeyDataType =
                            columnVectors[pKeyOrdinal].dataType();
                    Object pKey =
                            getPrimaryKeyValue(currRow, pKeyDataType,
                                    pKeyOrdinal);
                    if (!tableInfo.set.contains(pKey)) {
                        includeRow(currRow);
                        rowIdx++;
                    }
                }
            }
            columnarBatch2.setNumRows(rowIdx);
            if (rowIdx > 0) {
                return true;
            }
        } catch (InvalidRecordException e) {
            System.out.println("Muthu: PRIMARY KEY COLUMN NOT FOUND");
        }
        return false;
    }

  /**
   * Advances to the next batch of rows. Returns false if there are no more.
   */
  public boolean nextBatch() throws IOException {
      for (int i = 0; i < columnVectors.length; ++i) {
          columnVectors[i].reset();
          columnVectors2[i].reset();
      }
      columnarBatch.setNumRows(0);
      if (rowsReturned >= totalRowCount) {
          return false;
      }
      checkEndOfRowGroup();

      int num = (int) Math.min((long) capacity,
              totalCountLoadedSoFar - rowsReturned);
      for (int i = 0; i < columnReaders.length; ++i) {
          if (columnReaders[i] == null) {
              continue;
          }
          columnReaders[i].readBatch(num, columnVectors[i]);
      }
    rowsReturned += num;
    columnarBatch.setNumRows(num);
    numBatched = num;
    batchIdx = 0;
      return mergeBatch();
  }

  private void initializeInternal() throws IOException, UnsupportedOperationException {
    // Check that the requested schema is supported.
    missingColumns = new boolean[requestedSchema.getFieldCount()];
    List<ColumnDescriptor> columns = requestedSchema.getColumns();
    List<String[]> paths = requestedSchema.getPaths();
    for (int i = 0; i < requestedSchema.getFieldCount(); ++i) {
      Type t = requestedSchema.getFields().get(i);
      if (!t.isPrimitive() || t.isRepetition(Type.Repetition.REPEATED)) {
        throw new UnsupportedOperationException("Complex types not supported.");
      }

      String[] colPath = paths.get(i);
      if (fileSchema.containsPath(colPath)) {
        ColumnDescriptor fd = fileSchema.getColumnDescription(colPath);
        if (!fd.equals(columns.get(i))) {
          throw new UnsupportedOperationException("Schema evolution not supported.");
        }
        missingColumns[i] = false;
      } else {
        if (columns.get(i).getMaxDefinitionLevel() == 0) {
          // Column is missing in data but the required data is non-nullable. This file is invalid.
          throw new IOException("Required column is missing in data file. Col: " +
            Arrays.toString(colPath));
        }
        missingColumns[i] = true;
      }
    }
  }

  private void checkEndOfRowGroup() throws IOException {
    if (rowsReturned != totalCountLoadedSoFar) return;
    PageReadStore pages = reader.readNextRowGroup();
    if (pages == null) {
      throw new IOException("expecting more rows but reached last block. Read "
          + rowsReturned + " out of " + totalRowCount);
    }
    List<ColumnDescriptor> columns = requestedSchema.getColumns();
    List<Type> types = requestedSchema.asGroupType().getFields();
    columnReaders = new VectorizedColumnReader[columns.size()];
    for (int i = 0; i < columns.size(); ++i) {
      if (missingColumns[i]) continue;
      columnReaders[i] = new VectorizedColumnReader(
        columns.get(i),
        types.get(i).getOriginalType(),
        pages.getPageReader(columns.get(i)),
        convertTz,
        datetimeRebaseMode,
        int96RebaseMode);
    }
    totalCountLoadedSoFar += pages.getRowCount();
  }
}
