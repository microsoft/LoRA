/*

Copyright© 2006 by BBN Technologies and University of Maryland (UMD)

BBN and UMD grant a nonexclusive, source code, royalty-free right to
use this Software known as Translation Error Rate COMpute (the
"Software") solely for research purposes. Provided, you must agree
to abide by the license and terms stated herein. Title to the
Software and its documentation and all applicable copyrights, trade
secrets, patents and other intellectual rights in it are and remain
with BBN and UMD and shall not be used, revealed, disclosed in
marketing or advertisement or any other activity not explicitly
permitted in writing.

BBN and UMD make no representation about suitability of this
Software for any purposes.  It is provided "AS IS" without express
or implied warranties including (but not limited to) all implied
warranties of merchantability or fitness for a particular purpose.
In no event shall BBN or UMD be liable for any special, indirect or
consequential damages whatsoever resulting from loss of use, data or
profits, whether in an action of contract, negligence or other
tortuous action, arising out of or in connection with the use or
performance of this Software.

Without limitation of the foregoing, user agrees to commit no act
which, directly or indirectly, would violate any U.S. law,
regulation, or treaty, or any other international treaty or
agreement to which the United States adheres or with which the
United States complies, relating to the export or re-export of any
commodities, software, or technical data.  This Software is licensed
to you on the condition that upon completion you will cease to use
the Software and, on request of BBN and UMD, will destroy copies of
the Software in your possession.                                                

TERalignment.java v1
Matthew Snover (snover@cs.umd.edu)                           

*/

import java.lang.Math;
import java.util.HashMap;
import java.util.ArrayList;


/* Storage Class for TER alignments */
public class TERalignment {
  public String toString() {
	String s = "Original Ref: " + join(" ", ref) + 
      "\nOriginal Hyp: " + join(" ", hyp) + 
      "\nHyp After Shift: " + join(" ", aftershift);

	if (alignment != null) {
      s += "\nAlignment: (";
      for (int i = 0; i < alignment.length; i++) {s+=alignment[i];}
      s += ")";
	}
	if (allshifts == null) {
      s += "\nNumShifts: 0";
	} else {
      s += "\nNumShifts: " + allshifts.length;
      for (int i = 0; i < allshifts.length; i++) {
		s += "\n  " + allshifts[i];
      }
	}

    s += "\nScore: " + this.score() +
      " (" + this.numEdits + "/" + this.numWords + ")";
	return s;
  }

  public String toMoreString() {
	String s = "Best Ref: " + join(" ", ref) + 
      "\nOrig Hyp: " + join(" ", hyp) + "\n";
    s += getPraStr(ref, aftershift, alignment, allshifts, false);
    s += String.format("TER Score: %1$6.2f (%2$5.1f/%3$5.1f)\n", 100*this.score(),
                       this.numEdits, this.numWords);
    s += prtShift(ref, allshifts);
	return s;
  }

  public double score() { 
      if ((numWords <= 0.0) && (this.numEdits > 0.0)) { return 1.0; }
      if (numWords <= 0.0) { return 0.0; } 
      return (double) numEdits / numWords;
  }    

  public static String join(String delim, Comparable[] arr) {
	if (arr == null) return "";
	if (delim == null) delim = new String("");
	String s = new String("");
	for (int i = 0; i < arr.length; i++) {
      if (i == 0) { s += arr[i]; }
      else { s += delim + arr[i]; }
	}
	return s;
  }
    
  public static String join(String delim, char[] arr) {
	if (arr == null) return "";
	if (delim == null) delim = new String("");
	String s = new String("");
	for (int i = 0; i < arr.length; i++) {
      if (i == 0) { s += arr[i]; }
      else { s += delim + arr[i]; }
	}
	return s;
  }

  public void scoreDetails() {
	numIns = numDel = numSub = numWsf = numSft = 0;
	if(allshifts != null) {
      for(int i = 0; i < allshifts.length; ++i)
		numWsf += allshifts[i].size();
      numSft = allshifts.length;
	}
		
	if(alignment != null) {
      for(int i = 0; i < alignment.length; ++i) {
		switch (alignment[i]) {
          case 'S':
          case 'T':
		    numSub++;
		    break;
          case 'D':
		    numDel++;
		    break;
          case 'I':
		    numIns++;
		    break;
		}		
      }
	}
	//	if(numEdits != numSft + numDel + numIns + numSub)
	//      System.out.println("** Error, unmatch edit erros " + numEdits + 
	//                         " vs " + (numSft + numDel + numIns + numSub));
  }

  public static void performShiftArray(HashMap hwords,
                                       int start,
                                       int end,
                                       int moveto,
                                       int capacity) {
	HashMap nhwords = new HashMap();
	
	if(moveto == -1) {
      copyHashWords(hwords, nhwords, start, end, 0);
      copyHashWords(hwords, nhwords, 0, start - 1, end - start + 1);
      copyHashWords(hwords, nhwords, end + 1, capacity, end + 1);	    
	} else if (moveto < start) {
      copyHashWords(hwords, nhwords, 0, moveto, 0);
      copyHashWords(hwords, nhwords, start, end, moveto + 1);
      copyHashWords(hwords, nhwords, moveto + 1, start - 1, end - start + moveto + 2);
      copyHashWords(hwords, nhwords, end + 1, capacity, end + 1);
	} else if (moveto > end) {
      copyHashWords(hwords, nhwords, 0, start - 1, 0);
      copyHashWords(hwords, nhwords, end + 1, moveto, start);
      copyHashWords(hwords, nhwords, start, end, start + moveto - end);
      copyHashWords(hwords, nhwords, moveto + 1, capacity, moveto + 1);
	} else {
      copyHashWords(hwords, nhwords, 0, start - 1, 0);
      copyHashWords(hwords, nhwords, end + 1, end + moveto - start, start);
      copyHashWords(hwords, nhwords, start, end, moveto);
      copyHashWords(hwords, nhwords, end + moveto - start + 1, capacity, end + moveto - start + 1);
	}
	hwords.clear();
	hwords.putAll(nhwords);
  }

  private String prtShift(Comparable[] ref,
                          TERshift[] allshifts) {
	String to_return = "";
	int ostart, oend, odest;
	int nstart, nend;
	int dist;
	String direction = "";

	if(allshifts != null) {
      for(int i = 0; i < allshifts.length; ++i) {
        TERshift[] oneshift = new TERshift[1];
		ostart = allshifts[i].start;
		oend = allshifts[i].end;
		odest = allshifts[i].newloc;

		if(odest >= oend) {
          // right
          nstart = odest + 1 - allshifts[i].size();
          nend = nstart + allshifts[i].size() - 1;
          dist = odest - oend;
          direction = "right";
		} else {
          // left
          nstart = odest + 1;
          nend = nstart + allshifts[i].size() - 1;
          dist = ostart - odest -1;
          direction = "left";
		}

		to_return += "\nShift " + allshifts[i].shifted + " " + dist + " words " + direction;
		oneshift[0] = new TERshift(ostart, oend, allshifts[i].moveto, odest);
		to_return += getPraStr(ref, allshifts[i].aftershift, allshifts[i].alignment, oneshift, true); 
      }
      to_return += "\n";
	}
	return to_return;
  }

  private String getPraStr(Comparable[] ref,
                           Comparable[] aftershift,
                           char[] alignment,
                           TERshift[] allshifts,
                           boolean shiftonly) {
	String to_return = "";
	String rstr = "";
	String hstr = "";
	String estr = "";
	String sstr = "";
	HashMap align_info = new HashMap();
	ArrayList shift_dists = new ArrayList();
	int anum = 1;
	int ind_start = 0;
	int ind_end = 1;
	int ind_from = 2;
	int ind_in = 3;
	int ostart, oend, odest;
	int slen = 0;
	int nstart, nend, nfrom, dist;
	int non_inserr = 0;

	if(allshifts != null) {
      for(int i = 0; i < allshifts.length; ++i) {
		ostart = allshifts[i].start;
		oend = allshifts[i].end;
		odest = allshifts[i].newloc;
		slen = allshifts[i].size();

		if(odest >= oend) {
          // right
          nstart = odest + 1 - slen;
          nend = nstart + slen - 1;
          nfrom = ostart;
          dist = odest - oend;
		} else {
          // left
          nstart = odest + 1;
          nend = nstart + slen - 1;
          nfrom = ostart + slen;
          dist = (ostart - odest -1) * -1;
		}
	
		//dist = (allshifts[i].leftShift())?-1*allshifts[i].distance():allshifts[i].distance();
		shift_dists.add(dist);
		//		System.out.println("[" + hyp[ostart] + ".." + hyp[oend] + " are shifted " + dist);

		if(anum > 1) performShiftArray(align_info, ostart, oend, odest, alignment.length);

		Object val = align_info.get(nstart + "-" + ind_start);
		if(val == null) {
          ArrayList al = new ArrayList();
          al.add(anum);
          align_info.put(nstart + "-" + ind_start, al);
		} else {
          ArrayList al = (ArrayList) val;
          al.add(anum);
		}
		
		val = align_info.get(nend + "-" + ind_end);
		if(val == null) {
          ArrayList al = new ArrayList();
          al.add(anum);
          align_info.put(nend + "-" + ind_end, al);
		} else {
          ArrayList al = (ArrayList) val;
          al.add(anum);
		}
		
		val = align_info.get(nfrom + "-" + ind_from);
		if(val == null) {
          ArrayList al = new ArrayList();
          al.add(anum);
          align_info.put(nfrom + "-" + ind_from, al);
		} else {
          ArrayList al = (ArrayList) val;
          al.add(anum);
		}

		/*
          val = align_info.get("60-"+ind_start);
          if(val != null)
          System.out.println(((ArrayList) val).get(0));
          else
          System.out.println("empty");

          System.out.println("nstart: " + nstart + ", nend:" + nend + "," + ostart +"," + oend +","+ odest + "," + align_info.size());
		*/
		if(slen > 0) {
          for(int j = nstart; j <= nend; ++j) {
			val = align_info.get(j + "-" + ind_in);
			if(val == null) {
              ArrayList al = new ArrayList();
              al.add(anum);
              align_info.put(j + "-" + ind_in, al);
			} else {
              ArrayList al = (ArrayList) val;
              al.add(anum);
			}
          }
		}
		anum++;
      }
	}

	int hyp_idx = 0;
	int ref_idx = 0;
	Object val = null;
	if(alignment != null) {
      for(int i = 0; i < alignment.length; ++i) {
		String shift_in_str = "";
		String ref_wd = (ref_idx < ref.length)?String.valueOf(ref[ref_idx]):"";
		String hyp_wd = (hyp_idx < hyp.length)?String.valueOf(aftershift[hyp_idx]):"";
		int l = 0;

		if(alignment[i] != 'D') {
          val = align_info.get(hyp_idx + "-" + ind_from);
          if(val != null) {
			//						System.out.println("hyp_idx: " + hyp_idx + "," + hyp_wd);
			ArrayList list = (ArrayList) val;
			for(int j = 0; j < list.size(); ++j) {
              String s = "" + list.get(j);
              hstr += " @";
              rstr += "  ";
              estr += "  ";
              sstr += " " + s;
              for(int k = 1; k < s.length(); ++k) {
				hstr += " ";
				rstr += " ";
				estr += " ";
              }
			}
          }

          val = align_info.get(hyp_idx + "-" + ind_start);
          if(val != null) {
			//			System.out.println("hyp_idx: " + hyp_idx + "," + hyp_wd + "," + alignment.length);
			ArrayList list = (ArrayList) val;
			for(int j = 0; j < list.size(); ++j) {
              String s = "" + list.get(j);
              hstr += " [";
              rstr += "  ";
              estr += "  ";
              sstr += " " + s;
              for(int k = 1; k < s.length(); ++k) {
				hstr += " ";
				rstr += " ";
				estr += " ";
              }
			}
          }
          if(slen > 0) {
			val = align_info.get(hyp_idx + "-" + ind_in);
			if(val != null)
              shift_in_str = TERsgml.join(",", (ArrayList) val);
			//	if(val != null) System.out.println("shiftstr: " + ref_idx + "," + hyp_idx + "-" + ind_in + ":" + shift_in_str);
          } 
		}
		switch (alignment[i]) {
          case ' ':
		    l = Math.max(ref_wd.length(), hyp_wd.length());
		    hstr += " " + hyp_wd;
		    rstr += " " + ref_wd;
		    estr += " ";
		    sstr += " ";
		    for(int j = 0; j < l; ++j) {
              if(hyp_wd.length() <= j) hstr += " ";
              if(ref_wd.length() <= j) rstr += " ";
              estr += " ";
              sstr += " ";
		    }
		    hyp_idx++;
		    ref_idx++;
		    non_inserr++;
		    break;
          case 'S':
          case 'T':
		    l = Math.max(ref_wd.length(), Math.max(hyp_wd.length(), Math.max(1, shift_in_str.length())));
		    hstr += " " + hyp_wd;
		    rstr += " " + ref_wd;
		    if(hyp_wd.equals("") || ref_wd.equals("")) System.out.println("unexpected empty");
		    estr += " " + alignment[i];
		    sstr += " " + shift_in_str;
		    for(int j = 0; j < l; ++j) {
              if(hyp_wd.length() <= j) hstr += " ";
              if(ref_wd.length() <= j) rstr += " ";
              if(j > 0) estr += " ";
              if(j >= shift_in_str.length()) sstr += " ";
		    }
		    ref_idx++;
		    hyp_idx++;
		    non_inserr++;
		    break;
          case 'D':
		    l = ref_wd.length();
		    hstr += " ";
		    rstr += " " + ref_wd;
		    estr += " D";
		    sstr += " ";
		    for(int j = 0; j < l; ++j) {
              hstr += "*"; 
              if(j > 0) estr += " ";
              sstr += " ";
		    }
		    ref_idx++;
		    non_inserr++;
		    break;
          case 'I':
		    l = Math.max(hyp_wd.length(), shift_in_str.length());
		    hstr += " " + hyp_wd;
		    rstr += " ";
		    estr += " I";
		    sstr += " " + shift_in_str;
		    for(int j = 0; j < l; ++j) {
              rstr += "*"; 
              if(j >= hyp_wd.length()) hstr += " ";
              if(j > 0) estr += " ";
              if(j >= shift_in_str.length()) sstr += " ";
		    }
		    hyp_idx++;
		    break;
		}
		
		if(alignment[i] != 'D') {
          val = align_info.get((hyp_idx-1) + "-" + ind_end);
          if(val != null) {
			ArrayList list = (ArrayList) val;
			for(int j = 0; j < list.size(); ++j) {
              String s = "" + list.get(j);
              hstr += " ]";
              rstr += "  ";
              estr += "  ";
              sstr += " " + s;
              for(int k = 1; k < s.length(); ++k) {
				hstr += " ";
				rstr += " ";
				estr += " ";
              }
			}
          }		    
		}
      }
	}
	//	if(non_inserr != ref.length && ref.length > 1)
	    //      System.out.println("** Error, unmatch non-insertion erros " + non_inserr + 
	    //                         " and reference length " + ref.length );	
    String indent = "";
    if (shiftonly) indent = " ";
	to_return += "\n" + indent + "REF: " + rstr;
	to_return += "\n" + indent + "HYP: " + hstr;
    if(!shiftonly) {
      to_return += "\n" + indent + "EVAL:" + estr;
      to_return += "\n" + indent + "SHFT:" + sstr;
    }
	to_return += "\n";
	return to_return;
  }

  private static void copyHashWords(HashMap ohwords,
                                    HashMap nhwords,
                                    int start,
                                    int end,
                                    int nstart) {
	int ind_start = 0;
	int ind_end = 1;
	int ind_from = 2;
	int ind_in = 3;
	Object val = null;
	int k = nstart;

	for(int i = start; i <= end; ++k, ++i) {
      for(int j = ind_start; j <= ind_in; ++j) {
		val = ohwords.get(i + "-" + j);
		if(val != null) {
          ArrayList al = (ArrayList) val;		    
          nhwords.put(k + "-" + j, al);
		}
      }
	}
  }

  public Comparable[] ref;
  public Comparable[] hyp;
  public Comparable[] aftershift;

  public TERshift[] allshifts = null;

  public double numEdits = 0;
  public double numWords = 0.0;
  public char[] alignment = null;
  public String bestRef = "";

  public int numIns = 0;
  public int numDel = 0;
  public int numSub = 0;
  public int numSft = 0;
  public int numWsf = 0;

}
