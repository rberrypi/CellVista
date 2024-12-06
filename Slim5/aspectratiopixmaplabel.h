#ifndef ASPECTRATIOPIXMAPLABEL_H
#define ASPECTRATIOPIXMAPLABEL_H

#include <QLabel>
#include <QPixmap>
#include <QResizeEvent>

class AspectRatioPixmapLabel : public QLabel
{
	Q_OBJECT
public:
	explicit AspectRatioPixmapLabel(QWidget *parent = nullptr);
	virtual int heightForWidth(int width) const override;
	virtual QSize sizeHint() const override;
	QPixmap scaledPixmap() const;
public slots:
	void setPixmap(const QPixmap &);
	void resizeEvent(QResizeEvent *) override;
private:
	QPixmap pix;
};

#endif // ASPECTRATIOPIXMAPLABEL_H