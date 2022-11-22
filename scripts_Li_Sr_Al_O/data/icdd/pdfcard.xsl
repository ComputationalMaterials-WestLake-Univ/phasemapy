<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="pdfcard">
        <html>
            <body bgcolor="lightblue">
                <table border="0" width="75%" align="center">
                    <tr>
                        <td align="center">
                            <table border="0">
                                <tr>
                                    <td align="center">
                                        <h2>PDF Card</h2>
                                    </td>
                                </tr>
                            </table>
                            <table border="1">
                                <tr>
                                    <td>
                                        <b>PDF Number</b>
                                    </td>
                                    <td colspan="2">
                                        <xsl:value-of select="pdf_data/pdf_number" />
                                    </td>
                                    <td>
                                        <b>Status</b>
                                        <br></br>
                                        
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/status" />
                                    </td>
                                    <td colspan="2">
                                        <b>Quality Mark</b>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/quality_mark" />
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Environment</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/pressure_temperature" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Temperature</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/temperature" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Pressure</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/pressure" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Phase</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/phase" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Chemical Formula</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/chemical_formula" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Structural Formula</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/structural_formula" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Empirical Formula</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/empirical_formula" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
								<tr>
                                    <td>
                                        <b>Refined Formula</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/refined_formula" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Weight %</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/weight_percent" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Atomic %</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/atomic_percent" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Compound Name</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/chemical_name" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Mineral Name</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/mineral_name" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>IMA No.</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/ima_number" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Zeolite Name</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/zeolite_name" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Alternate Name</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/common_name" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>CAS Number</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/cas" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Entry Date</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/entry_date" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Modification Date</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/last_mod_date" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Modifications</b>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/last_modifications" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="4">
                                        <font size="4">
                                            <b>Experimental</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>Rad</b>
                                    </td>
                                    <td>
                                        <b>λ</b>
                                    </td>
                                    <td>
                                        <b>Filter</b>
                                    </td>
                                    <td>
                                        <b>d-Spacing</b>
                                    </td>
                                    <td>
                                        <b>Cutoff</b>
                                    </td>
                                    <td>
                                        <b>Intensity</b>
                                    </td>
                                    <td>
                                        <b>I/Ic</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/patrad" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/lambda" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/filttyp" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/instrument" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/lower" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/source" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/iic" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>I/Ic - ND</b>
                                    </td>
                                    <td>
                                        <b>Camera Diameter</b>
                                    </td>
                                    <td>
                                        <b>Internal Standard</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/iic_nd" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/diameter" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/internal_standard" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="13">
                                        <font size="4">
                                            <b>Physical</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>SYS</b>
                                    </td>
                                    <td>
                                        <b>
                                            <xsl:value-of select="pdf_data/spgr_label" />
                                        </b>
                                    </td>
                                    <td>
                                        <b>Aspect</b>
                                    </td>
                                    <td>
                                        <b>Modulation Wave Vectors</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xstal_system" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/spgr" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/spgraspect" />
                                        <br></br>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <xsl:for-each select="pdf_data/mod_wave_vectors/mod_wave_vector_row">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="mod_wave_vector"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Subsystems</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>ID</td>
                                                <td>W Matrix</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/subsystems/subsystem">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="subsystem_id"/>
                                                    </td>
                                                    <td>
                                                        <table border="1">
                                                            <xsl:for-each select="w_matrix/w_matrix_row">
                                                                <tr>
                                                                    <td>
                                                                        <xsl:value-of select="."/>
                                                                    </td>
                                                                </tr>
                                                            </xsl:for-each>
                                                        </table>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="4">
                                        <b>Author's Unit Cell</b>
                                    </td>
                                    <td>
                                        <b>a</b>
                                    </td>
                                    <td>
                                        <b>b</b>
                                    </td>
                                    <td>
                                        <b>c</b>
                                    </td>
                                    <td>
                                        <b>α</b>
                                    </td>
                                    <td>
                                        <b>β</b>
                                    </td>
                                    <td>
                                        <b>γ</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_a" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_b" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_c" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_alpha" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_beta" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_gamma" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Volume</b>
                                    </td>
                                    <td>
                                        <b>Z</b>
                                    </td>
                                    <td>
                                        <b>Molecular Volume/Formula Unit Volume</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_vol" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/z" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/cell_mv" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <b>Author's Unit Cell Axial Ratio</b>
                                    </td>
                                    <td>
                                        <b>c/a</b>
                                    </td>
                                    <td>
                                        <b>a/b</b>
                                    </td>
                                    <td>
                                        <b>c/b</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/authcovera" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/authaoverb" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/authcoverb" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <b>Density</b>
                                    </td>
                                    <td>
                                        <b>Dcalc</b>
                                    </td>
                                    <td>
                                        <b>Dmeas</b>
                                    </td>
                                    <td>
                                        <b>Dstruc</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtldx" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/dm" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ds" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>SS/FOM</b>
                                    </td>
                                    <td>
                                        <b>Melting Point</b>
                                    </td>
                                    <td>
                                        <b>R-factor</b>
                                    </td>
                                    <td>
                                        <b>Error</b>
                                    </td>
                                    <td>
                                        <b>Color</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/ss" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/melting_points" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/rfactor" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/error" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/color" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="13">
                                        <font size="4">
                                            <b>Crystal</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>Space Group</b>
                                    </td>
                                    <td>
                                        <b>Molecular Weight</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlsg" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlmolwt" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="4">
                                        <b>Crystal Data</b>
                                    </td>
                                    <td>
                                        <b>a</b>
                                    </td>
                                    <td>
                                        <b>b</b>
                                    </td>
                                    <td>
                                        <b>c</b>
                                    </td>
                                    <td>
                                        <b>α</b>
                                    </td>
                                    <td>
                                        <b>β</b>
                                    </td>
                                    <td>
                                        <b>γ</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtla" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlb" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlc" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlal" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlbe" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlga" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Volume</b>
                                    </td>
                                    <td>
                                        <b>Z</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlvol" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlz" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <b>Crystal Data Axial Ratio</b>
                                    </td>
                                    <td>
                                        <b>c/a</b>
                                    </td>
                                    <td>
                                        <b>a/b</b>
                                    </td>
                                    <td>
                                        <b>c/b</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlcovera" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlaoverb" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/xtlcoverb" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="5">
                                        <b>Reduced Cell</b>
                                    </td>
                                    <td>
                                        <b>a</b>
                                    </td>
                                    <td>
                                        <b>b</b>
                                    </td>
                                    <td>
                                        <b>c</b>
                                    </td>
                                    <td>
                                        <b>α</b>
                                    </td>
                                    <td>
                                        <b>β</b>
                                    </td>
                                    <td>
                                        <b>γ</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_a" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_b" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_c" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_al" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_be" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcell_ga" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Volume</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/redcellvol" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>
                                                    <xsl:value-of select="pdf_data/metric_symmetry_warning"/>
                                                </td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/metric_symmetry_warning_comments/metric_symmetry_warning_comment_group">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="metric_symmetry_warning_comment"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <font size="4">
                                            <b>Optical</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>εα</b>
                                    </td>
                                    <td>
                                        <b>πωβ</b>
                                    </td>
                                    <td>
                                        <b>εγ</b>
                                    </td>
                                    <td>
                                        <b>Sign</b>
                                    </td>
                                    <td>
                                        <b>2V</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/optical_data_alpha" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/optical_data_beta" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/optical_data_gamma" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/optical_data_sign" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/optical_data_2v" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="14">
                                        <font size="4">
                                            <b>Structure</b>
                                        </font>
                                    </td>
                                    <td colspan="7">
                                        <xsl:value-of select="pdf_data/structure_info"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>ADP Type</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/tf_type"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Origin</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/origin"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>SG Symmetry Operators</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Seq</td>
                                                <td>Operator</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/sg_sym_ops/sg_sym_op">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="seq"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="operator"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>SG Symmetry Operators (Modulated Structure)</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Seq</td>
                                                <td>Operator</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/sg_sym_ops_mod/sg_sym_op">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="seq"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="operator"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Atomic Coordinates</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Atom</td>
                                                <td>Num</td>
                                                <td>Wyckoff</td>
                                                <td>Symmetry</td>
                                                <td>x</td>
                                                <td>y</td>
                                                <td>z</td>
                                                <td>SOF</td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/idp_column"/>
                                                </td>
                                                <td>AET</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/atomic_coords/atomic_coord">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="atom"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="num"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="wyckoff"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="symmetry"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="x"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="y"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="z"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="sof"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="idp"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="aet"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Atomic Coordinates (Modulated)</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Atom</td>
                                                <td>Num</td>
                                                <td>Wyckoff</td>
                                                <td>x</td>
                                                <td>y</td>
                                                <td>z</td>
                                                <td>SOF</td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/idp_column"/>
                                                </td>
                                                <td>Subsys</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/atomic_coords_mod/atomic_coord">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="atom"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="num"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="wyckoff"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="x"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="y"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="z"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="sof"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="idp"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="subsys_id"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Anisotropic Displacement Parameters</b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Atom</td>
                                                <td>Num</td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    11
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    22
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    33
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    12
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    13
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    23
                                                </td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/ani_temp_facs/ani_temp_fac">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="atom"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="num"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp11"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp22"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp33"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp12"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp13"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp23"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Anisotropic Displacement Parameters (Modulated) </b>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Atom</td>
                                                <td>Num</td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    11
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    22
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    33
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    12
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    13
                                                </td>
                                                <td>
                                                    <xsl:value-of select="pdf_data/adp_column"/>
                                                    23
                                                </td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/ani_temp_facs_mod/ani_temp_fac">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="atom"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="num"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp11"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp22"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp33"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp12"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp13"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="adp23"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Crystal (Symmetry Allowed)</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/crystal_properties"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>AC Space Group</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/ac_space_group"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>AC Superspace Group</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/ac_superspace_group"/>
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <b>AC Unit Cell</b>
                                    </td>
                                    <td>
                                        <b>a</b>
                                    </td>
                                    <td>
                                        <b>b</b>
                                    </td>
                                    <td>
                                        <b>c</b>
                                    </td>
                                    <td>
                                        <b>α</b>
                                    </td>
                                    <td>
                                        <b>β</b>
                                    </td>
                                    <td>
                                        <b>γ</b>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_a" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_b" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_c" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_alpha" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_beta" />
                                        <br></br>
                                    </td>
                                    <td>
                                        <xsl:value-of select="pdf_data/ac_unit_cell_gamma" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="6">
                                        <font size="4">
                                            <b>Modulated</b>
                                        </font>
                                    </td>
                                    <tr>
                                        <td>
                                            <b>Positional Displacement Parameters</b>
                                        </td>
                                        <td colspan="6">
                                            <table border="1">
                                                <tr bgcolor="#9acd32">
                                                    <td>Atom Site</td>
                                                    <td>Axis</td>
                                                    <td>WV ID</td>
                                                    <td>Cos</td>
                                                    <td>Sin</td>
                                                </tr>
                                                <xsl:for-each select="pdf_data/positional_displacement_params/positional_displacement_param">
                                                    <tr>
                                                        <td>
                                                            <xsl:value-of select="atom_site"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="axis"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="wv_id"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="cos"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="sin"/>
                                                            <br></br>
                                                        </td>
                                                    </tr>
                                                </xsl:for-each>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <b>Atomic Displacement Parameters</b>
                                        </td>
                                        <td colspan="6">
                                            <table border="1">
                                                <tr bgcolor="#9acd32">
                                                    <td>Atom Site</td>
                                                    <td>WV ID</td>
                                                    <td>Cos</td>
                                                    <td>Sin</td>
                                                    <td>Tensor El.</td>
                                                </tr>
                                                <xsl:for-each select="pdf_data/atomic_displacement_params/atomic_displacement_param">
                                                    <tr>
                                                        <td>
                                                            <xsl:value-of select="atom_site"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="wv_id"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="cos"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="sin"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="tensor_element"/>
                                                            <br></br>
                                                        </td>
                                                    </tr>
                                                </xsl:for-each>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <b>Occupation Modulation</b>
                                        </td>
                                        <td colspan="6">
                                            <table border="1">
                                                <tr bgcolor="#9acd32">
                                                    <td>Atom Site</td>
                                                    <td>WV ID</td>
                                                    <td>Cos</td>
                                                    <td>Sin</td>
                                                </tr>
                                                <xsl:for-each select="pdf_data/occupation_modulations/occupation_modulation">
                                                    <tr>
                                                        <td>
                                                            <xsl:value-of select="atom_site"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="wv_id"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="cos"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="sin"/>
                                                            <br></br>
                                                        </td>
                                                    </tr>
                                                </xsl:for-each>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <b>Occupational - Crenel</b>
                                        </td>
                                        <td colspan="6">
                                            <table border="1">
                                                <tr bgcolor="#9acd32">
                                                    <td>Atom Site</td>
                                                    <td>C</td>
                                                    <td>W</td>
                                                </tr>
                                                <xsl:for-each select="pdf_data/crenel_functions/crenel_function">
                                                    <tr>
                                                        <td>
                                                            <xsl:value-of select="atom_site"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="crenel_c"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="crenel_w"/>
                                                            <br></br>
                                                        </td>
                                                    </tr>
                                                </xsl:for-each>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <b>Displacement - SawTooth</b>
                                        </td>
                                        <td colspan="6">
                                            <table border="1">
                                                <tr bgcolor="#9acd32">
                                                    <td>Atom Site</td>
                                                    <td>C</td>
                                                    <td>W</td>
                                                    <td>AX</td>
                                                    <td>AY</td>
                                                    <td>AZ</td>
                                                </tr>
                                                <xsl:for-each select="pdf_data/sawtooth_functions/sawtooth_function">
                                                    <tr>
                                                        <td>
                                                            <xsl:value-of select="atom_site"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="sawtooth_c"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="sawtooth_w"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="ax"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="ay"/>
                                                            <br></br>
                                                        </td>
                                                        <td>
                                                            <xsl:value-of select="az"/>
                                                            <br></br>
                                                        </td>
                                                    </tr>
                                                </xsl:for-each>
                                            </table>
                                        </td>
                                    </tr>
                                </tr>
                                <tr>
                                    <td rowspan="11">
                                        <font size="4">
                                            <b>Classifications</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>Subfiles</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/subfile_indicator" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Mineral Classification</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/pdf_minclass" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Zeolite Classification</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/pdf_zeoclass" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Pearson</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/pearson" />
                                        <br></br>
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Pearson w/o H</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/pearson_without_h" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Prototype Structure</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/rep_struc" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Prototype Structure (Alpha Sort)</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/rep_struc_alpha_sort" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>LPF Prototype Structure</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/lpf_rep_struc" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>LPF Prototype Structure (Alpha Sort)</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/lpf_rep_struc_alpha_sort" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>ANX</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/anx" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
								<tr>
                                    <td>
                                        <b>Wyckoff Sequence</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/wyckoff_sequence" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="2">
                                        <font size="4">
                                            <b>Cross-references</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>Cross-references</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/cross_ref_pdf_numbers" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Former PDF Numbers</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/former_pdf_number" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <font size="4">
                                            <b>References</b>
                                        </font>
                                    </td>
                                    <td colspan="6">
                                        <table border="1">
                                            <tr bgcolor="#9acd32">
                                                <td>Type</td>
                                                <td>DOI</td>
                                                <td>Reference</td>
                                            </tr>
                                            <xsl:for-each select="pdf_data/references/reference_group">
                                                <tr>
                                                    <td>
                                                        <xsl:value-of select="type"/>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:element name="a">
                                                            <xsl:attribute name="href">
                                                                <xsl:text>http://dx.doi.org/</xsl:text>
                                                                <xsl:value-of select="doi"/>
                                                            </xsl:attribute>
                                                            <xsl:value-of select="doi"/>
                                                        </xsl:element>
                                                        <br></br>
                                                    </td>
                                                    <td>
                                                        <xsl:value-of select="reference"/>
                                                        <br></br>
                                                    </td>
                                                </tr>
                                            </xsl:for-each>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td rowspan="3">
                                        <font size="4">
                                            <b>Comments</b>
                                        </font>
                                    </td>
                                    <td>
                                        <b>Database Comments</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/database_comments" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>User Comments</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/user_comments" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                                <tr>
                                    <td>
                                        <b>Shared Comments</b>
                                    </td>
                                    <td colspan="6">
                                        <xsl:value-of select="pdf_data/shared_comments" />
                                        <br></br>
                                        
                                    </td>
                                </tr>
                            </table>
                            <table border="0">
                                <tr>
                                    <td align="center">
                                        <h2>Diffraction Pattern</h2>
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center">
                                        <xsl:value-of select="graphs/wave_length" />
                                    </td>
                                </tr>
                            </table>
                            <center>
                                <img>
                                    <xsl:attribute name="src">
                                        <xsl:value-of select="graphs/chart_name" />
                                    </xsl:attribute>
                                </img>
                            </center>
                            <br></br>
                            <xsl:for-each select="graphs/stick_series">
                                <h2>
                                    <xsl:value-of select="@type"/>
                                </h2>
                                <table border="1">
                                    <tr bgcolor="#9acd32">
                                        <td>2θ</td>
                                        <td>TOF (µs)</td>
                                        <td>d(Å)</td>
                                        <td>Intensity</td>
                                        <td>h</td>
                                        <td>k</td>
                                        <td>l</td>
                                        <td>m</td>
                                        <td>n</td>
                                        <td>o</td>
                                        <td>*</td>
                                    </tr>
                                    <xsl:for-each select="intensity">
                                        <tr>
                                            <td>
                                                <xsl:value-of select="theta" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="tof" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="da" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="intensity" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="h" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="k" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="l" />
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="M1"/>
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="M2"/>
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <xsl:value-of select="M3"/>
                                                <br></br>
                                                
                                            </td>
                                            <td>
                                                <sup>
                                                    <xsl:value-of select="F" />
                                                </sup>
                                                <br></br>
                                                
                                            </td>
                                        </tr>
                                    </xsl:for-each>
                                </table>
                            </xsl:for-each>
                            <br></br>
                            <copy_right>© 2016 International Centre for Diffraction Data. All Rights Reserved</copy_right>
                        </td>
                    </tr>
                </table>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>