
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

TERcost.java v1
Matthew Snover (snover@cs.umd.edu)                           

*/


/* 

If you wish to experiment with other cost functions, (including
specific word to word cost matrices), a child class of TERcost should
be made, and an instance of it should be passed as the third arguement
to the TER function in TERcalc.

All costs must be in the range of 0.0 to 1.0.  Deviating outside of
this range may break the functionality of the TERcalc code.

This code does not do phrasal costs functions, such modification would require
changes to the TERcalc code.

In addition shifts only occur when the two subsequences are equal(),
and hash to the same location.  This can be modified from the standard
definition by using a new Comparable data structure which redefines
those functions.

*/

public class TERcost {
  /* For all of these functions, the score should be between 0 and 1
   * (inclusive).  If it isn't, then it will break TERcalc! */

  /* The cost of matching ref word for hyp word. (They are equal) */
  public double match_cost(Comparable hyp, Comparable ref) { 
      return _match_cost; 
  }

  /* The cost of substituting ref word for hyp word. (They are not equal) */
  public double substitute_cost(Comparable hyp, Comparable ref) {
	return _substitute_cost;
  }

  /* The cost of inserting the hyp word */
  public double insert_cost(Comparable hyp) {
	return _insert_cost;
  }

  /* The cost of deleting the ref word */
  public double delete_cost(Comparable ref) {
	return _delete_cost;
  }

  /* The cost of making a shift */
  public double shift_cost(TERshift shift) {
      return _shift_cost;
  }

    public double _shift_cost = 1.0;
    public double _insert_cost = 1.0;
    public double _delete_cost = 1.0;
    public double _substitute_cost = 1.0;
    public double _match_cost = 0.0;

}
