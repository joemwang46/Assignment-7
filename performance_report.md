### Syntax Differences**
- Read CSV | `pd.read_csv("file.csv")` | `pl.read_csv("file.csv")` 
- Compute returns | `df["price"].pct_change()` | `df.select(pl.col("price").pct_change())` 
- Drop missing values | `.dropna()` | `.drop_nulls()` 

### Performance Tradeoffs**
- Pandas is slower for large or wide datasets. Polars is much faster due to multithreading and SIMD
- Pandas has higher memory requirement due to Python object overhead

### GIL
- The **GIL** ensures only one thread executes Python bytecode at a time.  
- This design prevents race conditions that could corrupt shared data structures.  
- But also means threads cannot truly execute CPU-bound Python code simultaneously.

### **Threading**
- Lightweight; low startup cost.
- Shares memory between threads.
- Subject to the GIL — not suitable for CPU-bound computation.
**Best for:**
- **I/O-bound tasks**, e.g.:
  - Web scraping (waiting on network I/O)
  - Reading many files concurrently
  - Asynchronous user interfaces

### **Multiprocessing**
- True parallelism for CPU-bound workloads.  
- Higher memory overhead (each process has its own memory space).  
- Slower startup and serialization (pickling) cost.
**Best for:**
- **CPU-bound tasks**, e.g.:
  - Monte Carlo simulations  
  - Large numeric computations  
  - Matrix operations  
  - Long-running loops
