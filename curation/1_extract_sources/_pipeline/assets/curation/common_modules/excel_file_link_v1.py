"""Excel export target representation only; raw provenance is never changed."""
from urllib.parse import urlsplit, unquote
import re

def excel_hyperlink_target(target):
 """Use Unicode Windows paths for local file URIs containing Unicode.

 Excel interprets percent-encoded UTF-8 in local file URLs as legacy code-page
 bytes. Absolute Unicode paths avoid that decoding; page/cell fragments retain
 their exact existing text. ASCII file URLs and all other targets stay unchanged.
 """
 if not str(target).lower().startswith('file:'):return target
 parts=urlsplit(target);decoded=unquote(parts.path,encoding='utf-8',errors='strict')
 if not any(ord(c)>127 for c in decoded):return target
 if parts.netloc and parts.netloc.lower()!='localhost':
  path='\\\\'+unquote(parts.netloc,encoding='utf-8',errors='strict')+decoded.replace('/','\\')
 elif re.match(r'^/[A-Za-z]:/',decoded):path=decoded[1:].replace('/','\\')
 elif re.match(r'^[A-Za-z]:/',decoded):path=decoded.replace('/','\\')
 else:raise ValueError('Unicode file URI is not an absolute Windows path: '+target)
 if parts.query:path+='?'+parts.query
 if parts.fragment:path+='#'+parts.fragment
 return path

def physical_file_and_fragment(target):
 """Canonical read-only equivalence key for URI, Windows or bracketed Excel."""
 if str(target).lower().startswith('file:'):
  parts=urlsplit(target);p=unquote(parts.path,encoding='utf-8',errors='strict')
  if re.match(r'^/[A-Za-z]:/',p):p=p[1:]
  if parts.netloc and parts.netloc.lower()!='localhost':p='//'+parts.netloc+p
  return p.replace('/','\\'),parts.fragment
 return tuple(target.rsplit('#',1))if '#'in target else(target,'')
