import java.util.HashMap;
import java.util.ArrayList;
import java.util.regex.*;

public class TERpara {
  private static String reffn;
  private static String hypfn;
  private static String reflenfn;
  private static boolean mt_normalized;
  private static boolean case_on;
  private static boolean no_punctuation;
  private static ArrayList out_formats;
  private static String out_pfx;
  private static Pattern opts_p;
  private static int beam_width;
  private static String span_pfx;
  private static int shift_dist;

    private static double shift_cost;
    private static double delete_cost;
    private static double insert_cost;
    private static double substitute_cost;
    private static double match_cost;

  public static enum OPTIONS {
      NORMALIZE, CASEON, NOPUNCTUATION, REF, HYP, FORMATS, OUTPFX, BEAMWIDTH, REFLEN, TRANSSPAN, SHIFTDIST, DELETE_COST, INSERT_COST, SUBSTITUTE_COST, MATCH_COST, SHIFT_COST;
  }

  static {
	reffn = "";
	hypfn = "";
    reflenfn = "";
	out_pfx = "";
	mt_normalized = false;
	case_on = false;
	no_punctuation = false;
	opts_p = Pattern.compile("^\\s*-(\\S+)\\s*$");
	beam_width = 20;
    span_pfx = "";
    shift_dist = 50;
    shift_cost = 1.0;
    match_cost = 0.0;
    delete_cost = 1.0;
    insert_cost = 1.0;
    substitute_cost = 1.0;    
  }

  public static HashMap getOpts(String[] args) {
	HashMap paras = new HashMap();

	for(int i = 0; i < args.length; ++i) {
      Matcher m = opts_p.matcher(args[i]);

      if(m.matches()) {
		char opt = m.group(1).charAt(0);
		switch(opt) {
          case 'N':
		    mt_normalized = true;
		    break;
          case 's':
		    case_on = true;
		    break;
          case 'P':
		    no_punctuation = true;
		    break;
          case 'r':
		    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
		    else
              reffn = args[++i];
		    break;
          case 'h':
		    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
		    else
              hypfn = args[++i];
		    break;
          case 'o':
		    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
		    else
              out_formats = getOutFormats(args[++i]);
		    //		    System.out.println("formats: " + args[i]);
		    break;
          case 'n':
		    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
		    else
              out_pfx = args[++i];
		    break;
          case 'b':
		    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
		    else
              beam_width = Integer.valueOf(args[++i]);
		    break;
          case 'a':
            if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else
              reflenfn = args[++i];
            break;
          case 'S':
            if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else
              span_pfx = args[++i];
	    break;
	  case 'd':
            if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else
              shift_dist = Integer.valueOf(args[++i]);
	    break;
	  case 'M':
	    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else    
	      match_cost = Double.valueOf(args[++i]);
	    break;
	  case 'T':
	    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else    
	      shift_cost = Double.valueOf(args[++i]);
	    break;
	  case 'I':
	    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else    
	      insert_cost = Double.valueOf(args[++i]);
	    break;
	  case 'D':
	    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else    
	      delete_cost = Double.valueOf(args[++i]);
	    break;
	  case 'B':
	    if(i == args.length -1 || args[i+1].charAt(0) == '-')
              usage();
            else    
	      substitute_cost = Double.valueOf(args[++i]);
	    break;
          default:
		    ;
		}
      } else
		usage();
	}
	if(reffn.equals("") || hypfn.equals("")) {
      System.out.println("** Please specify both reference and hypothesis inputs");
      usage();
	} else {
      if (out_formats == null || out_formats.isEmpty())
		out_formats = getOutFormats("sum,pra,pra_more,xml,ter,sum_nbest");
	    
      paras.put(OPTIONS.NORMALIZE, mt_normalized);
      paras.put(OPTIONS.CASEON, case_on);
      paras.put(OPTIONS.NOPUNCTUATION, no_punctuation);
      paras.put(OPTIONS.OUTPFX, out_pfx);
      paras.put(OPTIONS.REF, reffn);
      paras.put(OPTIONS.HYP, hypfn);
      paras.put(OPTIONS.FORMATS, out_formats);
      paras.put(OPTIONS.BEAMWIDTH, beam_width);
      paras.put(OPTIONS.REFLEN, reflenfn);
      paras.put(OPTIONS.TRANSSPAN, span_pfx);
      paras.put(OPTIONS.SHIFTDIST, shift_dist);
      paras.put(OPTIONS.SHIFT_COST, shift_cost);
      paras.put(OPTIONS.MATCH_COST, match_cost);
      paras.put(OPTIONS.INSERT_COST, insert_cost);
      paras.put(OPTIONS.DELETE_COST, delete_cost);
      paras.put(OPTIONS.SUBSTITUTE_COST, substitute_cost);
	}
	return paras;
  }

  public static void usage() {
	System.out.println("** Usage: java -jar tercom.jar [-N] [-s] [-P] -r ref -h hyp [-a alter_ref] [-b beam_width] [-S trans_span_prefix] [-o out_format -n out_pefix] [-d max_shift_distance] [-M match_cost] [-D delete_cost] [-B substitute_cost] [-I insert_cost] [-T shift_cost]");
	System.exit(1);
  }

  private static ArrayList getOutFormats(String s) {
	ArrayList ret = new ArrayList();
	String [] arrays = s.split(",");
	if(arrays != null)
      for(int i = 0; i < arrays.length; ++i)
		ret.add(arrays[i]);
	return ret;
  }

}
