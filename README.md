# meic2wrfchem
Interpolating & distributing MEIC 0.25*0.25 emission inventory onto WRF-Chem grids, with many customizable distribution coefficients
*** 
本程序针对[清华大学MEIC源排放清单](http://meicmodel.org/)(0.25°*0.25°，2016)，实现了清单中`CB05`排放源向`WRF-Chem`模式`CBMZ-Mosaic`方案的插值和分配（包括日分配、周分配、高度层分配等）。   


本程序主要受成都信息工程大学樊晋老师的[meic2wrf](https://github.com/jinfan0931/meic2wrf)项目的启发，与樊老师项目的主要区别在于：   
1. 实现的是基于`MEIC`模型的`CB05`排放源，制作`WRF-Chem`模式的`CBMZ-Mosaic`化学方案。对应`WRF-Chem`参数可参考`namelist.input`文件;
2. 提供了各排放源的日分配、周分配、高度分配系数接口，可自由修改；
3. 生成的是逐小时排放源（`io_style_emissions=2`）。由于包含了日分配和周分配，因而生成的逐小时排放源存在周期变化，并非如`io_style_emissions=1`一样固定不变。
4. 使用`multiprocessing`模块，支持并行生成排放源文件。

最后，再次感谢樊晋老师([@jinfan0931](https://github.com/jinfan0931))[meic2wrf](https://github.com/jinfan0931/meic2wrf)项目的启发；感谢高超老师([@gc13141112](https://github.com/gc13141112))提供的各类排放源周变化系数。