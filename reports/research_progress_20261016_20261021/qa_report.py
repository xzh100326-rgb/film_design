from pathlib import Path
from zipfile import ZipFile
from urllib.parse import unquote, urlparse
import json
import xml.etree.ElementTree as ET
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw

root=Path(__file__).resolve().parent
qa=root/'qa'
pdf=qa/'research_record.pdf'
reader=PdfReader(str(pdf))
render=pdfium.PdfDocument(str(pdf))
pages=[]
thumbnails=[]
for i, page in enumerate(render):
    image=page.render(scale=2).to_pil().convert('RGB')
    image.save(qa/f'page-{i+1}.png')
    thumb=ImageOps.contain(image,(290,410))
    tile=Image.new('RGB',(310,445),'#d9d9d9')
    tile.paste(thumb,((310-thumb.width)//2,14))
    ImageDraw.Draw(tile).text((12,425),f'Page {i+1}',fill='black')
    thumbnails.append(tile)
    text=reader.pages[i].extract_text() or ''
    pages.append({'page':i+1,'characters':len(text),'start':text[:95], 'end':text[-95:]})
columns=3
rows=(len(thumbnails)+columns-1)//columns
contact=Image.new('RGB',(columns*310,rows*445),'white')
for i,tile in enumerate(thumbnails): contact.paste(tile,((i%columns)*310,(i//columns)*445))
contact.save(qa/'contact_sheet.jpg',quality=90)
docx=next(root.glob('*.docx'))
with ZipFile(docx) as z:
    rels=ET.fromstring(z.read('word/_rels/document.xml.rels'))
    links=[]
    for item in rels:
        if item.attrib.get('Type','').endswith('/hyperlink'):
            target=item.attrib['Target']
            parsed=urlparse(target)
            path=unquote(parsed.path).lstrip('/') if parsed.scheme=='file' else None
            links.append({'target':target, 'exists':Path(path).exists() if path else None})
    image_count=len([n for n in z.namelist() if n.startswith('word/media/')])
audit={'pages':pages, 'embedded_image_count':image_count,'hyperlinks':links,
       'all_local_hyperlinks_exist':all(x['exists'] for x in links)}
(qa/'structural_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'page_count':len(pages),'image_count':image_count,'hyperlinks':len(links),
                  'all_hyperlinks_exist':audit['all_local_hyperlinks_exist'],'pages':pages},ensure_ascii=False,indent=2))
